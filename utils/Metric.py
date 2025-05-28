import torch


class Metrics(object):
    def __init__(self, args):
        super().__init__()
        self.PAD = 0
        self.k_list = args.metric_k

    def compute_metric(self, y_prob, y_true):
        # Get valid samples (non-PAD)
        valid_mask = y_true != self.PAD
        y_prob_valid = y_prob[valid_mask]
        y_true_valid = y_true[valid_mask]
        scores_len = valid_mask.sum().item()

        # Get top-k predictions for all samples at once
        top_k_values, top_k_indices = torch.topk(y_prob_valid, k=max(self.k_list), dim=1)
        
        scores = {}
        for k in self.k_list:
            # Calculate Hit@K
            top_k = top_k_indices[:, :k]
            hit_at_k = (top_k == y_true_valid.unsqueeze(1)).any(dim=1)
            scores[f'hit@{k}'] = hit_at_k.float().mean().item()
            
            # Calculate MAP@K
            map_at_k = self._compute_map_at_k(y_true_valid, top_k_indices, k)
            scores[f'map@{k}'] = map_at_k

        return scores, scores_len

    def _compute_map_at_k(self, y_true, top_k_indices, k):
        """Computes MAP@K using PyTorch operations."""
        # Create a mask for correct predictions
        correct_mask = (top_k_indices[:, :k] == y_true.unsqueeze(1))
        
        # Calculate precision at each position
        precision_at_k = correct_mask.float()
        
        # Calculate cumulative sum of correct predictions
        cumsum = torch.cumsum(correct_mask, dim=1)
        
        # Calculate average precision
        ap = (precision_at_k * cumsum) / torch.arange(1, k + 1, device=y_true.device)
        
        # Sum up average precision for each sample
        ap = ap.sum(dim=1) / torch.clamp(cumsum[:, -1], min=1)
        
        return ap.mean().item()


def apk(actual, predicted, k):
    """ Computes the average precision at K. """
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / min(len(actual), k)
