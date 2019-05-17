import argparse
import sys

import cbir
import cbir.commands
import cbir.configuration

base_parser = argparse.ArgumentParser(add_help=False, prog='CBIRCore')
base_parser.add_argument('-c', '--config',
                         help='Path to config')
base_parser.add_argument('-l', '--log_prefix',
                         help='Log prefix')


def create_parser():
    """Creates ArgumentParser to parse command-line options"""

    parser = argparse.ArgumentParser(description='CBIRCore commands',
                                     parents=[base_parser])

    subparsers = parser.add_subparsers(title='Commands', dest='command')

    prepare_directory_structure_parser = subparsers.add_parser('prepare_cbir_directory_structure', help='Prepare directory structure')
    prepare_directory_structure_parser.add_argument('--persistent_state', required=False,
                                                    help="Root directory for cbir's persistent_state")

    evaluate_with_all_descriptors_parser = subparsers.add_parser('evaluate_with_all_descriptors',
                                                                 help='Evaluate CBIRCore pipeline with all descriptors')
    evaluate_with_all_descriptors_parser.add_argument('--train_dir', default=None)
    evaluate_with_all_descriptors_parser.add_argument('--test_dir', default=None)
    evaluate_with_all_descriptors_parser.add_argument('--gt_dir', default=None)
    evaluate_with_all_descriptors_parser.add_argument('--is_sample', action='store_true', default=False, help='Whether not to use all data')

    evaluate_parser = subparsers.add_parser('evaluate',
                                            help='Evaluate CBIRCore pipeline chosen descriptor')
    evaluate_parser.add_argument('--train_dir', default=None)
    evaluate_parser.add_argument('--test_dir', default=None)
    evaluate_parser.add_argument('--gt_dir', default=None)
    evaluate_parser.add_argument('--is_sample', action='store_true',
                                 default=False, help='Whether not to use all data')
    evaluate_parser.add_argument('--des_type', required=True)
    evaluate_parser.add_argument('--sv', action='store_true', default=False)
    evaluate_parser.add_argument('--qe', action='store_true', default=False)

    evaluate_only_parser = subparsers.add_parser('evaluate_only',
                                                 help='Evaluate CBIRCore pipeline witn already built index')
    evaluate_only_parser.add_argument('database_name')
    evaluate_only_parser.add_argument('index_name')
    evaluate_only_parser.add_argument('database_photos_dir')
    evaluate_only_parser.add_argument('gt_dir')
    evaluate_only_parser.add_argument('--sv', action='store_true', default=False)
    evaluate_only_parser.add_argument('--qe', action='store_true', default=False)
    evaluate_only_parser.add_argument('--p_fine_max', type=float, required=False, default=None)

    create_index_parser = subparsers.add_parser('create_index',
                                                 help='Create CBIR Index')
    create_index_parser.add_argument('database_name')
    create_index_parser.add_argument('index_name')
    create_index_parser.add_argument('path_to_images')
    create_index_parser.add_argument('max_images', type=int)
    create_index_parser.add_argument('--des_type', required=False)
    create_index_parser.add_argument('--max_keypoints', type=int, required=False)
    create_index_parser.add_argument('--K', type=int, required=True)
    create_index_parser.add_argument('--L', type=int, required=True)

    change_params_parser = subparsers.add_parser('change_params',
                                                 help='Change K, L and maybe other params')
    change_params_parser.add_argument('database_name')
    change_params_parser.add_argument('index_name')
    change_params_parser.add_argument('--des_type', required=False)
    change_params_parser.add_argument('--max_keypoints', type=int, required=False)
    change_params_parser.add_argument('--K', type=int, required=True)
    change_params_parser.add_argument('--L', type=int, required=True)

    search_parser = subparsers.add_parser('search',
                                          help='Search')
    search_parser.add_argument('database_name')
    search_parser.add_argument('index_name')
    search_parser.add_argument('query')
    search_parser.add_argument('--n_candidates', type=int, required=False)
    search_parser.add_argument('--topk', type=int, required=False)
    search_parser.add_argument('--sv', action='store_true', default=False)
    search_parser.add_argument('--qe', action='store_true', default=False)
    search_parser.add_argument('--p_fine_max', type=float, required=False, default=None)

    return parser


def main(argv):
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        raise SystemExit
    command = getattr(cbir.commands, args.command)
    if not command:
        parser.print_help()
        raise SystemExit

    # config_path = args.config
    # config = cbir.configuration.make_config(config_path)
    cbir.configuration.configure_logging(log_level='INFO', prefix=args.log_prefix)
    command(**vars(args))


if __name__ == '__main__':
    main(sys.argv[1:])
