from typing import List, Dict, Any
import yaml
import os

def _all_hypotheses_resolved(hypotheses: List[Dict[str, Any]]) -> bool:
    if len(hypotheses) == 0:
        return False
    for h in hypotheses:
        status = h.get('status')
        if status not in ['supported', 'rejected', 'needs_revision']:
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

def save_checkpoint(state: dict, node_name: str, checkpoint_dir: str = "checkpoints"):
    """
    stateをyamlで保存する。ノード名ごとにファイルを分ける。
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"{node_name}.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(state, f, allow_unicode=True)
    return path

def load_checkpoint(node_name: str, checkpoint_dir: str = "checkpoints"):
    """
    yamlからstateを復元する。
    """
    path = os.path.join(checkpoint_dir, f"{node_name}.yaml")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        state = yaml.safe_load(f)
    return state 