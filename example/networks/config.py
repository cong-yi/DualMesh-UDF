import argparse
import pprint


## This code refers the code in NGLOD (https://github.com/nv-tlabs/nglod)

def parse_options(return_parser=False):

    parser = argparse.ArgumentParser(description='Train deep implicit 3D geometry representations.')
    
    ## global information
    global_group = parser.add_argument_group('global')
    global_group.add_argument('--mesh_prefix', type = str, default =  'example/results/',
                              help='Path for saving the meshes')

    # Architecture for network
    net_group = parser.add_argument_group('net')
    net_group.add_argument('--init_dims', type = int, nargs = 8, default = [512, 512, 512, 512, 512, 512, 512, 512])
    net_group.add_argument('--output_dims', type = int, default = 1)
    net_group.add_argument('--dropout',type = int, nargs = 8, default = [])
    net_group.add_argument('--dropout_prob', type = float, default = 0)
    net_group.add_argument('--init_latent_in', type = int, nargs = 8, default = [])
    net_group.add_argument('--init_norm_layers', type = int, nargs = 8, default = [])
    net_group.add_argument('--weight_norm', action = 'store_true', help = 'Apply weight norm to the layers.')
    net_group.add_argument('--xyz_in_all', action = 'store_true')
    net_group.add_argument('--latent_dropout', action = 'store_true')
    net_group.add_argument('--use_pe', action = 'store_true')
    net_group.add_argument('--pe_dimen', type = int, default = 6)
    net_group.add_argument('--activation', type = str , default = 'sine', choices = ['sine', 'relu', 'softplus'])
    net_group.add_argument('--last_activation', type = str , default = 'softplus', choices = ['relu', 'softplus'])
    net_group.add_argument('--pretrained',type=str, default=None,
                            help = 'The checkpoint that we want to load.')

    # Parse and run
    if return_parser:
        return parser
    else:
        return argparse_to_str(parser)


def argparse_to_str(parser):

    args = parser.parse_args()

    args_dict = {}
    for group in parser._action_groups:
        group_dict = {a.dest:getattr(args, a.dest, None) for a in group._group_actions}
        args_dict[group.title] = vars(argparse.Namespace(**group_dict))

    pp = pprint.PrettyPrinter(indent=2)
    args_str = pp.pformat(args_dict)
    args_str = f'```{args_str}```'

    return args, args_str

