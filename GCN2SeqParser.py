import argparse
def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', type=bool, default=False, help='whether in debug mode')
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:4')
    parser.add_argument('--comment', type=str,
                        default='lstm')
    parser.add_argument('--dataset', type=str, default='SYDNEY')
    parser.add_argument('--seed', type=int, default=2019)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr_init', type=float, default=0.005)
    parser.add_argument('--lr_scheduler_step', type=int, default=50)
    parser.add_argument('--lr_scheduler_rate', type=float, default=0.3)
    parser.add_argument('--cuda', action='store_false', help='enables cuda')

    parser.add_argument('--dim', type=int, default=1)
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--window', type=int, default=12)
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--valdays', type=int, default=10)
    parser.add_argument('--testdays', type=int, default=10)
    return parser

if __name__ == '__main__':
    param = get_parser().parse_args()
    print(param.seed)
    print(param.device)
    print(param.cuda)
    pass