from typing import List, Dict, Any

def _all_hypotheses_resolved(hypotheses: List[Dict[str, Any]]) -> bool:
    if not hypotheses:
        return False
    for h in hypotheses:
        status = h.get('status')
        # supported でも verification が未完了なら未解決とみなす
        if status == 'supported':
            verif = h.get('verification', {}) if isinstance(h, dict) else {}
            if verif.get('status') not in ['verified', 'disproved']:
                return False
        elif status not in ['rejected', 'needs_revision']:
            # new / inconclusive / investigating など
            return False
    return True

def _unresolved_ratio(hypotheses: List[Dict[str, Any]]) -> float:
    if not hypotheses:
        return 0.0
    unresolved: List[Dict[str, Any]] = []
    for h in hypotheses:
        status = h.get('status')
        if status in ['rejected']:
            continue
        if status == 'supported':
            verif = h.get('verification', {}) if isinstance(h, dict) else {}
            if verif.get('status') not in ['verified', 'disproved']:
                unresolved.append(h)
        else:
            unresolved.append(h)
    return len(unresolved) / len(hypotheses)

def _count_inconclusive(hypotheses: List[Dict[str, Any]]) -> int:
    return sum(1 for h in hypotheses if h['status'] in ['inconclusive', 'needs_revision'])

def _summarize_hypotheses(hypotheses: List[Dict[str, Any]]):
    return [
        {
            "id": h.get("id"),
            "status": h.get("status"),
            "priority": h.get("priority"),
        }
        for h in hypotheses
    ] 