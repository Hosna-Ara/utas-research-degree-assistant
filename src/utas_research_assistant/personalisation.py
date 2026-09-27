"""Explainable ranking and session context over the existing hybrid corpus."""
import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from utas_research_assistant.graph.queries import get_projects_by_supervisor, get_supervisor_profile, canonical_supervisor_name

from utas_research_assistant.applicant import ApplicantProfile, VOCAB
from utas_research_assistant.retrieval.corpus import tokenize
from utas_research_assistant.retrieval.filters import canonical
from utas_research_assistant.generation.models import AnswerResponse

STOP = set('the and for with from that this research experience skills interests university bachelor master degree science using in of to a an on my'.split())
WEIGHTS = {'Research/topic': 30, 'Skills/methods': 25, 'Academic/domain': 15, 'Supervisor fit': 15, 'Practical fit': 15}


def terms(values):
    return {t for t in tokenize(' '.join(values)) if t not in STOP and (len(t) > 2 or t in {'r', 'ai'})}


def overlap(values, text):
    wanted = terms(values)
    found = wanted & set(tokenize(text))
    return sorted(found), len(found) / len(wanted) if wanted else None


def score_project(profile, item):
    m = item.metadata
    supervisor = m.get('supervisor_profile') or {}
    groups = {
        'Research/topic': (profile.values('research_interests', 'preferred_research_areas'), str(m.get('description') or item.text)),
        'Skills/methods': (profile.values('technical_skills', 'programming_tools', 'methods'), str(m.get('description') or item.text)),
        'Academic/domain': (profile.values('academic_disciplines', 'domain_experience'), item.text),
        'Supervisor fit': (profile.values('research_interests', 'preferred_research_areas', 'academic_disciplines'), ' '.join(supervisor.get('research_fields') or [])),
    }
    breakdown = {}
    strengths = []
    for label, (values, evidence) in groups.items():
        matches, ratio = overlap(values, evidence)
        supported = bool(evidence and terms(values))
        if label == 'Skills/methods':
            source_methods = '|'.join(VOCAB[k] for k in ('technical_skills', 'programming_tools', 'methods')).split('|')
            supported = supported and (bool(matches) or any(re.search(r'(?<!\w)' + re.escape(t) + r'(?!\w)', evidence, re.I) for t in source_methods))
        breakdown[label] = {'weight': WEIGHTS[label], 'points': round(WEIGHTS[label] * (ratio or 0), 1) if supported else None, 'overlaps': matches, 'evidence': evidence[:1600] if supported else ''}
        strengths.extend(matches)
    checks = []
    requested_checks = 0
    unknowns = []
    for pf, mf in [('location_preferences', 'location'), ('degree_preference', 'degree_types'), ('student_type', 'student_types'), ('funding_preference', 'funding_status')]:
        preferences = profile.values(pf)
        value = m.get(mf)
        if not preferences:
            continue
        requested_checks += 1
        if not value or value == 'unknown':
            unknowns.append(f'{mf.replace("_", " ")}: not evidenced in the project record')
            continue
        actual = ' '.join(value) if isinstance(value, list) else str(value)
        if pf == 'funding_preference':
            preference = ' '.join(preferences).lower()
            if re.search(r'no funding|self.fund|not required|no preference', preference):
                unknowns.append('Funding preference does not request a stipend; no funding constraint scored.')
                requested_checks -= 1
                continue
            if not re.search(r'fund|scholarship|stipend', preference):
                unknowns.append('Funding preference could not be interpreted reliably.')
                continue
            matched = value == 'funded'
        else:
            field_name = {'location_preferences': 'location', 'degree_preference': 'degree_type', 'student_type': 'student_type'}[pf]
            actual_values = value if isinstance(value, list) else actual.split(';')
            expected = [canonical(field_name, p) for p in preferences]
            known = [canonical(field_name, v) for v in actual_values]
            matched = any(p == v or re.search(r'(?<!\w)' + re.escape(v) + r'(?!\w)', p) for p in expected for v in known)
        checks.append((pf, actual, bool(matched)))
    practical_weight = 15 * len(checks) / requested_checks if requested_checks else 0
    breakdown['Practical fit'] = {'weight': 15, 'supported_weight': practical_weight, 'points': round(practical_weight * sum(c[2] for c in checks) / len(checks), 1) if checks else None, 'overlaps': [f'{k}: {v}' for k, v, ok in checks if ok], 'evidence': str(checks)}
    coverage = sum(v.get('supported_weight', v['weight']) for v in breakdown.values() if v['points'] is not None)
    score = round(100 * sum(v['points'] or 0 for v in breakdown.values()) / coverage) if coverage else 0
    gaps = [f'{k.replace("_", " ")}: preference does not match recorded {v}; confirm with UTAS.' for k, v, ok in checks if not ok]
    # Mention only source-evidenced methods, never turn absence into an applicant deficiency.
    skills = '|'.join(VOCAB[k] for k in ('technical_skills', 'programming_tools', 'methods')).split('|')
    absent = [t for t in skills if re.search(r'(?<!\w)' + re.escape(t) + r'(?!\w)', str(m.get('description') or ''), re.I) and not re.search(r'(?<!\w)' + re.escape(t) + r'(?!\w)', ' '.join(profile.values(*profile.facts)), re.I)]
    if absent:
        unknowns.append('Project mentions ' + ', '.join(absent[:5]) + '; your profile does not explicitly evidence these. This is not proof of a skills gap or a formal requirement.')
    unknowns.extend(f'{k}: unknown (missing applicant or source evidence)' for k, v in breakdown.items() if v['points'] is None)
    return {'score': score, 'coverage': round(coverage, 1), 'breakdown': breakdown, 'strengths': sorted(set(strengths)), 'gaps': gaps, 'unknowns': unknowns}


@dataclass
class ConversationContext:
    project_ids: list[str] = field(default_factory=list)
    supervisor_ids: list[str] = field(default_factory=list)
    active_project: str | None = None
    active_supervisor: str | None = None
    stage: str = ''
    last_question: str = ''


def suggestions(question='', payload=None, context=None, has_profile=False):
    context = context or ConversationContext()
    q = question.lower()
    if 'fund' in q or 'scholarship' in q:
        result = ['Which matched projects are funded?' if context.project_ids else 'Find funded PhD projects', 'What are the international applicant requirements?', 'What should I prepare before applying?', 'What scholarships does UTAS offer?']
    elif any(t in q for t in ('english', 'entry requirement', 'prepare', 'applying')):
        result = ['What English tests does UTAS accept?', 'What scholarships does UTAS offer?', 'How do I find a research supervisor?', 'Show Master by Research projects in Hobart']
    elif context.stage == 'supervisor':
        result = ["Show this supervisor's research interests", 'Which projects does this supervisor lead?', 'How should I contact this supervisor?', 'What should I mention from my CV?' if has_profile else 'What are the PhD entry requirements?']
        if 'research interests' in q:
            result = ['What should I mention from my CV?' if has_profile else 'How should I contact this supervisor?', 'Draft talking points for contacting them', 'Show related projects', 'Compare my profile with this supervisor' if has_profile else 'What are the PhD entry requirements?']
    elif context.stage == 'projects' and context.project_ids:
        result = ['Show me the supervisor for Project 1', 'Compare these projects against my profile' if has_profile else 'What funding is available for Project 1?', 'What gaps do I have for Project 1?' if has_profile else 'Tell me more about Project 1', 'Which project best matches my skills?' if has_profile else 'What should I prepare before applying?']
    elif has_profile:
        result = ['Find the 3 UTAS research projects most aligned with my CV', 'Which supervisors are most aligned with my CV?', 'Find 5 projects that match my skills and research interests', 'What are the PhD entry requirements?']
    elif 'english' in q or 'entry' in q or 'apply' in q:
        result = ['What English tests does UTAS accept?', 'What scholarships does UTAS offer?', 'How do I find a research supervisor?', 'Show Master by Research projects in Hobart']
    else:
        result = ['Find AI and machine learning PhD projects', 'What are the PhD entry requirements?', 'Find funded ICT projects for international students', 'Which supervisors work in artificial intelligence?']
    # Do not repeat the question just answered.
    result = [s for s in dict.fromkeys(result) if s.lower().rstrip('?.') != q.rstrip('?.')]
    if len(result) < 4:
        extra = 'Show related projects' if context.active_supervisor else 'What should I prepare before applying?'
        if extra not in result:
            result.append(extra)
    return result[:4]


class Personalisation:
    """Stateless service: all applicant and entity state belongs to the caller session."""
    def __init__(self, service):
        self.service = service
        self.retriever = service.router.retriever
        self.projects = {str(i.metadata['project_id']): i for i in self.retriever.corpus.items if i.item_type == 'research_project'}
        self.supervisors = {str(i.metadata['supervisor_id']): i for i in self.retriever.corpus.items if i.item_type == 'supervisor_profile'}
        # Some graph supervisor identities have no downloaded discovery profile.
        # Keep the evidenced identity usable while marking research fields unknown.
        known = {canonical_supervisor_name(i.metadata['canonical_name']) for i in self.supervisors.values()}
        for pid, project in self.projects.items():
            name = project.metadata.get('primary_supervisor')
            if not name or canonical_supervisor_name(name) in known:
                continue
            graph_profile = get_supervisor_profile(service.router.graph, name) or {}
            sid = 'graph-supervisor-' + re.sub(r'[^a-z0-9]+', '-', canonical_supervisor_name(name))
            related = [p for p, item in self.projects.items() if canonical_supervisor_name(item.metadata.get('primary_supervisor') or '') == canonical_supervisor_name(name)]
            metadata = {**graph_profile, 'supervisor_id': sid, 'canonical_name': name, 'title': name,
                        'research_fields': graph_profile.get('research_fields') or [], 'related_project_ids': related,
                        'source_url': graph_profile.get('source_url') or project.metadata['source_url'],
                        'profile_unavailable': not bool(graph_profile.get('discovery_url'))}
            self.supervisors[sid] = SimpleNamespace(metadata=metadata, text='Supervisor: ' + name)
            known.add(canonical_supervisor_name(name))

    def rank(self, profile, limit=5):
        query = ' '.join(profile.values('research_interests', 'preferred_research_areas', 'technical_skills', 'programming_tools', 'methods', 'academic_disciplines'))
        if not query.strip():
            return []
        # Existing semantic/BM25/RRF supplies candidate order and deterministic tie breaks.
        retrieved = self.retriever.search(query, top_k=len(self.projects), scope='projects')
        order = {str(r['project_id']): n for n, r in enumerate(retrieved)}
        rows = [{'project_id': pid, **score_project(profile, item)} for pid, item in self.projects.items()]
        rows.sort(key=lambda r: (-r['score'], -r['coverage'], order.get(r['project_id'], 9999), r['project_id']))
        return rows[:limit]

    def score_supervisor(self, profile, item):
        related = [self.projects[p].text for p in item.metadata.get('related_project_ids', []) if p in self.projects]
        proxy = SimpleNamespace(metadata={'description': item.text + '\n' + '\n'.join(related)}, text=item.text)
        score = score_project(profile, proxy)
        return score

    def rank_supervisors(self, profile, limit=5):
        rows = [{'supervisor_id': sid, **self.score_supervisor(profile, item)} for sid, item in self.supervisors.items()]
        query = ' '.join(profile.values('research_interests', 'technical_skills', 'methods', 'academic_disciplines'))
        retrieved = self.retriever.search(query, top_k=len(self.supervisors), scope='supervisors') if query else []
        order = {r['supervisor_id']: n for n, r in enumerate(retrieved)}
        rows.sort(key=lambda r: (-r['score'], -r['coverage'], order.get(r['supervisor_id'], 9999), r['supervisor_id']))
        return rows[:limit] if terms(profile.values(*profile.facts)) else []

    def supervisor_for(self, pid):
        project = self.projects.get(pid)
        if not project:
            return None
        name = project.metadata.get('primary_supervisor')
        for sid, item in self.supervisors.items():
            if pid in item.metadata.get('related_project_ids', []) or item.metadata.get('canonical_name') == name:
                return sid
        return None

    def resolve(self, question, context):
        q = question.lower()
        ids = []
        for pid, item in self.projects.items():
            if re.search(r'(?<!\d)' + re.escape(pid) + r'(?!\d)', q) or item.metadata['title'].lower() in q:
                ids.append(pid)
        for value in re.findall(r'project\s+(\d+)', q):
            n = int(value)
            if value in self.projects:
                if value not in ids:
                    ids.append(value)
            elif 1 <= n <= len(context.project_ids):
                ids.append(context.project_ids[n - 1])
        if not ids and 'first project' in q and context.project_ids:
            ids = context.project_ids[:1]
        if not ids and re.search(r'this project|the project', q) and context.active_project:
            ids = [context.active_project]
        if not ids and (('compare' in q and 'supervisor' not in q) or 'matched projects' in q or re.search(r'these(?: [3-5])? projects', q)):
            ids = context.project_ids[:5]
        if ids:
            context.active_project = ids[0]
            context.active_supervisor = self.supervisor_for(ids[0])
        sm = re.search(r'(?:supervisor\s+(\d+)|the (first|second|third) supervisor)', q)
        if sm:
            n = int(sm[1]) if sm[1] else ['first', 'second', 'third'].index(sm[2]) + 1
            if n <= len(context.supervisor_ids):
                context.active_supervisor = context.supervisor_ids[n - 1]
        return list(dict.fromkeys(ids))

    def answer(self, question, profile, context):
        q = question.lower()
        context.last_question = question
        # Preserve the established private-document route.
        if 'private' in q or 'local document' in q or 'applicant b' in q:
            return None
        personal = bool(re.search(r'\bcv\b|resume|my (?:profile|background|skills|research interests)|suitable for me|gaps|aligned', q))
        if not personal and not profile and not context.project_ids and not context.active_supervisor and not re.search(r'project\s+[1-9]\b|this project|this supervisor|first supervisor', q):
            return None  # Preserve ordinary named-entity questions in the established router.
        ids = self.resolve(question, context)
        follow = bool(ids or (context.active_supervisor and re.search(r'this supervisor|first supervisor|second supervisor|third supervisor|contact|mention|talking points|related projects', q)))
        if not personal and not follow:
            if re.search(r'project\s+\d+|this project|this supervisor|first supervisor', q):
                return self._result(question, 'I cannot resolve that reference in this conversation. Ask for recommendations or give the project title or supervisor name.', [], [], [])
            return None
        if personal and profile is None:
            return self._result(question, 'Upload a PDF or DOCX CV in the sidebar CV / Profile area, then click Save CV to enable personalised matching.', [], [], [])
        if re.search(r'project\s+[1-9]\b', q) and not ids:
            return self._result(question, 'That project number is not in the recent recommendations. Ask for matches first or use a project title or ID.', [], [], [])
        count = re.search(r'\b([3-5])\b', q)
        limit = int(count[1]) if count else 5
        if 'compare' in q and 'supervisor' in q and context.supervisor_ids:
            context.stage = 'supervisor'
            return self._supervisor_answer(question, context.supervisor_ids[:limit], profile, context)
        if follow and context.active_supervisor and re.search(r'supervisor|contact|mention|talking points|related projects', q):
            sid = context.active_supervisor
            context.stage = 'supervisor'
            if re.search(r'which.*projects|related projects|other projects', q):
                related = self.related_projects(sid)
                context.project_ids = related
                context.active_project = related[0] if related else None
                return self._project_answer(question, related[:5], profile)
            return self._supervisor_answer(question, [sid], profile, context)
        if personal and 'supervisor' in q and not ids:
            sids = [r['supervisor_id'] for r in self.rank_supervisors(profile, limit)]
            context.supervisor_ids = sids
            context.active_supervisor = sids[0] if sids else None
            context.stage = 'supervisor'
            context.active_project = None
            return self._supervisor_answer(question, sids, profile, context)
        if not ids and 'this project' in q:
            return self._result(question, 'Please select a project first; the supervisor may have more than one project.', [], [], [])
        if not ids:
            ranked = self.rank(profile, limit) if profile else []
            ids = [r['project_id'] for r in ranked]
            context.project_ids = ids
            context.active_project = ids[0] if ids else None
            context.active_supervisor = self.supervisor_for(ids[0]) if ids else None
        context.stage = 'projects'
        if 'funded' in q and ids:
            ids = [pid for pid in ids if self.projects[pid].metadata.get('funding_status') == 'funded']
        return self._project_answer(question, ids, profile)

    def _project_answer(self, question, ids, profile):
        rows = []
        parts = ['Scores are reproducible keyword alignment indicators, not admission probabilities. Missing dimensions are excluded; coverage is the supported weight out of 100. Source snapshot: September 2026; verify availability with UTAS.']
        sources = []
        for n, pid in enumerate(ids, 1):
            item = self.projects[pid]
            m = item.metadata
            score = score_project(profile, item) if profile else None
            citation = f'S{n}'
            sources.append({'citation_id': citation, 'item_type': item.item_type, **m, 'text': item.text})
            if score:
                rows.append({'project_id': pid, **score})
            part = f'### #{n} {m["title"]}\nProject {pid} · {m.get("primary_supervisor") or "Supervisor not recorded"}\n\n'
            if score:
                part += f'**Alignment: {score["score"]}% · Evidence coverage: {score["coverage"]}%**\n\n'
                reasons = [f'{label}: your profile and the source both mention {", ".join(d["overlaps"])}.' for label, d in score['breakdown'].items() if d['overlaps'] and label != 'Practical fit']
                part += '**Why it matches:**\n\n' + ('\n'.join(f'- {reason} [{citation}]' for reason in reasons) or 'No explicit keyword overlaps found; this is a weak match.') + '\n\n'
                part += '**Strengths:** ' + (', '.join(score['strengths']) or 'Not evidenced') + '\n\n'
                part += '**Potential gaps / unknowns:** ' + ' '.join(score['gaps'] + score['unknowns'] or ['No additional gap evidenced. Eligibility and suitability still require supervisor review.']) + '\n\n'
                part += '\n'.join(f'- {label}: {d["points"]}/{d["weight"]}' if d['points'] is not None else f'- {label}: unknown / {d["weight"]}' for label, d in score['breakdown'].items()) + '\n\n'
            part += f'Field: {", ".join(m.get("research_categories") or []) or "Not recorded"}. Location: {m.get("location") or "Unknown"}. Funding: {m.get("scholarship_text") or "Unknown"}. Degree: {", ".join(m.get("degree_types") or []) or "Unknown"}. Student types: {", ".join(m.get("student_types") or []) or "Unknown"}. [{citation}]'
            part += f'\n\n[Official project]({m["source_url"]})'
            supervisor = m.get('supervisor_profile') or {}
            if supervisor.get('discovery_url') or supervisor.get('source_url'):
                part += f' · [Supervisor profile]({supervisor.get("discovery_url") or supervisor["source_url"]})'
            parts.append(part)
        if not ids:
            parts = ['No projects matched this request with the available evidence. Save a more detailed CV or broaden the constraints.']
        if 'compare' in question.lower() and len(rows) > 1:
            strongest = max(rows, key=lambda r: r['score'])
            parts.insert(1, f'**Comparison:** Project {strongest["project_id"]} has the highest supported alignment in this set ({strongest["score"]}%, coverage {strongest["coverage"]}%). Compare coverage and the individual overlaps below before deciding; scores do not establish eligibility.')
        return self._result(question, '\n\n'.join(parts), sources, ids, rows)

    def related_projects(self, sid):
        m = self.supervisors[sid].metadata
        graph_rows = get_projects_by_supervisor(self.service.router.graph, m['canonical_name'])
        related = [str(r['project_id']) for r in graph_rows if str(r['project_id']) in self.projects]
        return related or [pid for pid in m.get('related_project_ids', []) if pid in self.projects]

    def _supervisor_answer(self, question, sids, profile, context):
        parts, sources = [], []
        for n, sid in enumerate(sids, 1):
            sup = self.supervisors[sid]
            m = sup.metadata
            related = self.related_projects(sid)
            matches, ratio = overlap(profile.values('research_interests', 'technical_skills', 'academic_disciplines') if profile else [], ' '.join(m.get('research_fields') or []) + sup.text)
            citation = f'S{n}'
            sources.append({'citation_id': citation, 'item_type': 'supervisor_profile', **m, 'text': sup.text})
            parts.append(f'### #{n} {m["canonical_name"]}\n\nResearch interests: {", ".join(m.get("research_fields") or []) or "Not recorded"}. [{citation}]\n\n' + ('**Applicant overlaps:** ' + (', '.join(matches) or 'No explicit overlap evidenced') + '.\n\n' if profile else '') + 'Relevant projects: ' + ('; '.join(f'[{pid}: {self.projects[pid].metadata["title"]}]({self.projects[pid].metadata["source_url"]})' for pid in related) or 'Not recorded') + f'. [{citation}]\n\n[Official source]({m["source_url"]})')
            if m.get('profile_unavailable'):
                parts.append('A resolved supervisor discovery profile is unavailable in this snapshot. The name and project relationship are evidenced; additional research interests and contact details are unknown.')
            if profile:
                sr = self.score_supervisor(profile, sup)
                parts.append(f'**Supervisor alignment: {sr["score"]}% · Evidence coverage: {sr["coverage"]}%**. Normalised over topic (30), skills/methods (25), and academic/domain (15); practical suitability is unknown. No publication metrics are used.\n\n' + '; '.join(f'{k}: {v["points"]}/{v["weight"]}' for k, v in sr['breakdown'].items()) + '\n\nPotential gaps / unknowns: a missing profile overlap is not evidence of an applicant deficiency. Confirm supervision availability and eligibility directly.')
            if re.search(r'contact|approach|mention|talking points', question, re.I):
                parts.append('Suggested talking points (not a claim about supervisor availability): introduce your degree and research aim; discuss ' + (', '.join(matches[:5]) or 'the listed research themes') + '; reference the project title and ask about suitability, availability and funding. Use the official profile contact details. Do not claim experience absent from your profile.')
        if not parts:
            parts = ['No supervisor relationship was evidenced for these matches.']
        result = self._result(question, '\n\n'.join(parts), sources, [], [])
        result.evidence['graph_operation_used'] = 'projects_by_supervisor'
        result.evidence['supervisor_ids'] = sids
        result.response.tool_used = 'SPARQL Knowledge Graph'
        result.response.tool_result = 'Supervisor–project relationships verified against the local graph.'
        return result

    @staticmethod
    def _result(question, answer, sources, ids, rows):
        from utas_research_assistant.service import AnswerResult
        response = AnswerResponse(question=question, answer=answer, reasoning_method='profile_matching', planner_method='deterministic', citations=[s['citation_id'] for s in sources], sources=sources, insufficient_evidence=not sources, project_ids=ids, generation_model='deterministic')
        return AnswerResult(response, {'ranked_retrieval_evidence': sources, 'recommendations': rows})
