import argparse
import sys

import cbir
import cbir.commands
import cbir.configuration

base_parser = argparse.ArgumentParser(add_help=False, prog='CBIR')
base_parser.add_argument('-c', '--config',
                         help='Path to config')


def create_parser():
    """Creates ArgumentParser to parse command-line options"""

    parser = argparse.ArgumentParser(description='CBIR commands',
                                     parents=[base_parser])

    # The command selected by the user will be stored in `args.command`
    subparsers = parser.add_subparsers(title='Commands', dest='command')

    # TODO: Add config parameter and do not require some parameters here.
    register_parser = subparsers.add_parser('register', help='Register new database of images')
    register_parser.add_argument('database', help='Name of a new registered database')
    register_parser.add_argument('path', help='Path to directory containing images')

    add_images_parser = subparsers.add_parser('add_images', help='Add images to database.'
                                                                 'Does not perform complete recomputations.')
    add_images_parser.add_argument('database', help='Name of a database to add images to')
    add_images_parser.add_argument('path', help='Path to directory containing images to add')

    reindex_database_parser = subparsers.add_parser('reindex_database',
                                                    help='Compute descriptors for new previously added images'
                                                         'and rebuild bow and iverted_index.')
    reindex_database_parser.add_argument('database', help='Name of a database to reindex')

    search_parser = subparsers.add_parser('search', help='Search similar images in database')
    search_parser.add_argument('database', help='Name of a database where to search')
    search_parser.add_argument('query', help='Path to an image to search similar for')
    search_parser.add_argument('--save', action='store_true', default=False, help='Whether to store query image and result')
    search_parser.add_argument('--tag', '-t', default=None, help='Tag for distinguishing between search sessions')

    show_parser = subparsers.add_parser('show', help='Show session from sessions history')
    show_parser.add_argument('database', help='Name of a database related to session')
    show_parser.add_argument('tag', help='Tag of a session to show')

    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate CBIR pipeline')
    evaluate_parser.add_argument('--sample', action='store_true', default=False, help='Whether not to use all data')
    evaluate_parser.add_argument('--train_dir', default=None)
    evaluate_parser.add_argument('--test_dir', default=None)
    evaluate_parser.add_argument('--gt_dir', default=None)

    prepare_directory_structure_parser = subparsers.add_parser('prepare_directory_structure', help='Prepare directory structure')
    prepare_directory_structure_parser.add_argument('--persistent_state', required=False,
                                                    help="Root directory for cbir's persistent_state")

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

    config_path = args.config
    cbir.configuration.configure(config_path, args)
    command(args)


if __name__ == '__main__':
    main(sys.argv[1:])
