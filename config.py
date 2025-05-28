import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="The clean implementation of SILN.")

    parser.add_argument("--seed", type=int, default=21, help="Random seed")
    parser.add_argument('--dropout', type=float, default=0.3, help="dropout ratio")

    # ========================= Data Configs ==========================
    parser.add_argument('--dataset', type=str, default='christian')
    parser.add_argument('--batch_size', default=32, type=int, help="batch_size use for training duration")
    parser.add_argument('--val_batch_size', default=32, type=int, help="batch_size use for validation duration")
    parser.add_argument('--test_batch_size', default=32, type=int, help="batch_size use for testing duration")
    parser.add_argument('--max_len', type=int, default=200, help="The maximum length of cascade")

    # ========================= Learning Configs ==========================
    parser.add_argument('--max_epochs', type=int, default=50, help="How many epochs")
    parser.add_argument('--print_steps', type=int, default=10, help="Number of steps to log training metrics.")
    parser.add_argument('--learning_rate', default=5e-4, type=float, help="Initial learning rate")

    # ======================== HyperParameter Configs =========================
    parser.add_argument('--sample_k', type=int, default=100, help="The sampling number of neighbors")
    parser.add_argument('--window_size', type=int, default=3, help="The size of current stage, i.e., q")
    parser.add_argument('--dim', type=int, default=64, help="The dimension of embeddings d")
    parser.add_argument('--n_heads', type=int, default=6, help="The number of attention heads B")
    parser.add_argument('--metric_k', type=list, default=[10, 50, 100])

    # ======================== SavedModel Configs =========================
    parser.add_argument('--saved_model_path', type=str, default='checkpoint/')
    parser.add_argument('--ckpt_file', type=str, default='checkpoint/model_.bin')
    parser.add_argument('--best_score', type=float, default=0., help='save checkpoint if hits+map > best_score')

    return parser.parse_args()
