"""Conservative, image-bound acquisition rules; absence never negates a route."""
import re

CLASSES = {'MRI': r'\b(?:MRI|MR|magnetic resonance)\b',
           'CT': r'\b(?:CT|comput(?:ed|er) tomography)\b',
           'XR': r'\b(?:X[- ]?ray|radiograph)\b',
           'US': r'\b(?:ultrasound|ultrasonograph\w*)\b'}
ROUTES = {'IV': r'\b(?:IV|intravenous)\b', 'ORAL': r'\b(?:oral|GI|gastrointestinal)\b',
          'INTRATHECAL': r'\bintrathecal\b', 'RECTAL': r'\brectal\b'}

def contrast_state(text):
    text = text.lower()
    negative = bool(re.search(r'\b(?:non[- ]?contrast|unenhanced|without (?:any )?contrast)\b', text))
    positive_text = re.sub(r'non[- ]?contrast|without (?:any )?contrast|unenhanced', '', text)
    positive = bool(re.search(r'\bcontrast(?:[- ]enhanced)?\b|\benhanced\b', positive_text))
    routes = {}
    for route, pattern in ROUTES.items():
        if re.search(pattern, text, re.I):
            routes[route] = 'NO' if re.search(r'(?:without|no|non[- ]?)\s*(?:\w+\s+){0,1}'+pattern, text, re.I) else 'YES'
    # "Without IV contrast" is route-specific, not a claim about oral contrast.
    route_negative = bool(re.search(r'\b(?:without|no)\s+(?:iv|intravenous|oral|gi|rectal|intrathecal)\s+contrast', text))
    if route_negative:
        positive_text = re.sub(r'\b(?:without|no)\s+(?:iv|intravenous|oral|gi|rectal|intrathecal)\s+contrast', '', positive_text)
        positive = bool(re.search(r'\bcontrast\b|\benhanced\b', positive_text))
    state = 'MIXED' if negative and positive else 'NONCONTRAST' if negative else 'CONTRAST_WITH_QUALIFIERS' if positive else 'UNKNOWN'
    return dict(state=state, routes=routes, scope='UNQUALIFIED_NONCONTRAST' if negative else 'SPECIFIED_ONLY', evidence=text)

def ct_relation(source, target):
    if source['state'] == 'MIXED' or target['state'] == 'MIXED': return 'UNKNOWN'
    for route, value in target['routes'].items():
        if route in source['routes'] and source['routes'][route] != value: return 'CONTRADICTS'
    if {source['state'],target['state']}=={'NONCONTRAST','CONTRAST_WITH_QUALIFIERS'}:
        positive=source if source['state']=='CONTRAST_WITH_QUALIFIERS' else target
        # Noncontrast acquisition is not a blanket negation of oral/rectal/etc.
        # A route-specific positive without IV enhancement remains unresolved.
        if positive['routes'] and positive['routes'].get('IV')!='YES':return 'UNKNOWN'
        return 'CONTRADICTS'
    if source['state'] == target['state'] == 'NONCONTRAST': return 'COMPATIBLE'
    if source['state'] == target['state'] == 'CONTRAST_WITH_QUALIFIERS':
        return 'COMPATIBLE' if source['routes'] == target['routes'] else 'UNKNOWN'
    return 'UNKNOWN'

def image_acquisition(source):
    """Only current-image attribution, or an explicit operator-checked panel map.

    panel_binding is evidence metadata, never inferred from a composite boolean.
    A complete panel map requires a recorded image inspection and exact spans.
    """
    text = source['source_text']
    binding = source.get('panel_binding')
    if binding:
        if binding.get('inspection') != 'IMAGE_LABELS_CHECKED': return None, None, 'PANEL_INSPECTION_MISSING'
        panels = binding['panels']; selected = source.get('bound_panel')
        keys = [selected] if selected else binding['image_labels']
        if not keys or any(k not in panels for k in keys): return None, None, 'PANEL_MAP_INCOMPLETE'
        entries = [panels[k] for k in keys]
        if any(not e['span'] or e['span'] not in text for e in entries): return None, None, 'PANEL_SPAN_UNBOUND'
        modalities = {e['modality'] for e in entries}
        statuses = [contrast_state(e['span']) for e in entries]
        if len(modalities) != 1: return None, None, 'MIXED_MODALITY_PANELS'
        if len({(s['state'],tuple(sorted(s['routes'].items()))) for s in statuses}) != 1:
            return next(iter(modalities)), dict(state='MIXED',routes={},scope='PANEL_SET',evidence=[e['span'] for e in entries]), 'MIXED_CONTRAST_PANELS'
        return next(iter(modalities)), statuses[0], 'EXPLICIT_COMPLETE_PANEL_MAP' if not selected else 'EXPLICIT_SELECTED_PANEL'
    # No automatic universal "all panels" statement. Multi-panel or temporally
    # compared captions need a panel map even if the first modality looks clear.
    panels = re.search(r'\([a-i](?:\s*(?:and|[-–,])\s*[a-i])?\)|\b[a-i]\s*[,–-]\s*[a-i]\b|\b(?:panel|panels)\b|(?:^|[.;])\s*[A-I]:',text,re.I)
    history = re.search(r'\b(?:previous|prior|historical|follow[- ]up|after \d|before|compared with|comparison with)\b',text,re.I)
    if source.get('composite') or panels: return None, None, 'PANEL_BINDING_UNRESOLVED'
    if history: return None, None, 'TEMPORAL_OR_COMPARISON_BINDING_UNRESOLVED'
    first = text.strip().split('.')[0]
    classes = {k for k,p in CLASSES.items() if re.search(p,first,re.I)}
    if len(classes)!=1: return None, None, 'CURRENT_IMAGE_MODALITY_UNRESOLVED'
    name = next(iter(classes)); match = re.search(CLASSES[name],first,re.I)
    prefix = first[:match.start()]
    if re.search(r'\b(?:not|no|without|other|another|subsequent|previous|prior|unlike)\b',prefix,re.I):
        return None, None, 'MODALITY_MENTION_NOT_CURRENT_POSITIVE_ATTRIBUTION'
    if re.search(CLASSES[name]+r'\s*(?:scan|imaging|examination)?\s*(?:(?:was|is|were)\s+)?(?:not|never)\s*(?:done|performed|obtained|available|used|shown)',first,re.I) or re.search(r'\b(?:possibly|possible|recommended|may have|could have)\b',first,re.I):
        return None, None, 'MODALITY_NEGATED_OR_UNCERTAIN_ACQUISITION'
    # Require a caption that begins with its acquisition, not discussion of one.
    if len(prefix.split())>8 or not re.match(r'^[\w\s()–-]*$',prefix):
        return None, None, 'CURRENT_IMAGE_ATTRIBUTION_UNRESOLVED'
    return name, contrast_state(first) if name=='CT' else None, 'CURRENT_IMAGE_CAPTION_ATTRIBUTION'

def automatic_decision(edit, source):
    if source.get('evidence_origin')=='LLM_DERIVED': return None
    op=edit['question_operator']
    if op not in {'modality','ct_contrast'}: return None
    if source['source_dataset']!='PMC_PRIMARY': return None
    m, state, binding=image_acquisition(source)
    if op=='ct_contrast':
        target=contrast_state(edit['target_answer'])
        if m!='CT' or state is None:
            observed=contrast_state(source['source_text'])
            state=dict(state='MIXED' if observed['state']=='MIXED' else 'UNKNOWN',routes={},scope='UNBOUND_MENTIONS',evidence=source['source_text'])
            return dict(level='UNKNOWN',relation='UNKNOWN',reason=binding,answer=None,state=state,target=target)
        relation=ct_relation(state,target)
        answer='Non-contrast CT' if state['state']=='NONCONTRAST' else state['evidence']
        # Second compatibility check after answer construction.
        if not isinstance(answer,str) or ct_relation(contrast_state(answer),target)!=relation:
            relation='UNKNOWN';binding='GENERATED_ANSWER_COMPATIBILITY_UNRESOLVED'
        return dict(level='SOURCE_VERIFIED' if relation=='CONTRADICTS' else 'REJECTED' if relation=='COMPATIBLE' else 'UNKNOWN',relation=relation,reason=binding,answer=answer,state=state,target=target)
    target_classes={k for k,p in CLASSES.items() if re.search(p,edit['target_answer'],re.I)}
    relation='CONTRADICTS' if m and len(target_classes)==1 and m not in target_classes else 'COMPATIBLE' if m in target_classes else 'UNKNOWN'
    return dict(level='SOURCE_VERIFIED' if relation=='CONTRADICTS' else 'REJECTED' if relation=='COMPATIBLE' else 'UNKNOWN',relation=relation,reason=binding,answer={'XR':'X-ray','US':'Ultrasound'}.get(m,m),state=None)
