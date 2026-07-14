import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="training parameters")

    parser.add_argument("--inner_loop_lr", type=float, default=1e-3, help="learner learning rate.", )
    parser.add_argument("--inner_loop_momentum", type=float, default=0.9, help="learner learning momentum.", )
    parser.add_argument("--test_batch_size", type=int, default=100, )
    parser.add_argument("--val_batch_size", type=int, default=100, )
    parser.add_argument("--inner_batch_size", type=int, default=256, help="learner training batch size", )
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--epoches", type=int, default=20, help="learner training epoch")
    parser.add_argument("--per_gpu_popsize", type=int, default=10, help="GA pop size: GPU num * per_gpu_popsize")
    parser.add_argument("--maxiter", type=int, default=80, help="GA max iter")
    parser.add_argument("--log_period", type=int, default=2)
    parser.add_argument("--split", type=int, default=5, help="test batch size=total test data num / split")
    parser.add_argument("--scale", type=int, default=0, help="Diffusion guidance scale,XL-Turbo please use 0")
    parser.add_argument("--img_shape", nargs='+', type=int, default=[224, 224])
    parser.add_argument("--prompt_nums_per_class", type=int, default=20)
    parser.add_argument("--use_real_img", action='store_true', default=False)
    parser.add_argument("--inner_loop_optimizer", type=str, default="Adam")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_id", type=str,
                        default='stabilityai/sdxl-turbo')
    parser.add_argument("--dataset", type=str, default='')
    parser.add_argument("--test_annotation_path", type=str, default='')
    parser.add_argument("--val_annotation_path", type=str, default='')
    parser.add_argument("--label_path", type=str, default='')
    parser.add_argument("--root_dir", type=str, default='')
    parser.add_argument("--learner_type", type=str, default="clip_head")
    parser.add_argument("--meta_loss_type", type=str, default='average')
    parser.add_argument("--save_dir", type=str, default='')
    parser.add_argument("--label_path_to_opt", type=str, default=None)
    parser.add_argument("--pool_path", type=str, default='')
    parser.add_argument("--syn_path", type=str, default='', help="if not use_real_img, we don't use this arg")
    parser.add_argument("--train_annotation_path", type=str, default='',
                        help="if not use_real_img, we don't use this arg")
    parser.add_argument("--model_path", type=str, default='')

    # bellow is useless args, will be removed
    parser.add_argument("--separate", action='store_true', default=False)
    parser.add_argument("--common_prompt_nums", type=int, default=0)
    parser.add_argument("--compute_method", action='store_true')
    parser.add_argument("--learned_dict", type=str, default='')
    parser.add_argument("--clip_clean", action='store_true')
    parser.add_argument("--validation_learner_type", type=str, default=None)
    parser.add_argument("--resize", action='store_true', default=True)
    parser.add_argument("--guidance_scale_learn", action='store_true')
    parser.add_argument("--enable_checkpointing", action='store_true', default=True)
    parser.add_argument("--step_by_step_validation", action='store_true', default=True)
    parser.add_argument("--compute_baseline", action='store_true')
    parser.add_argument("--generator_batch", type=int, default=200)

    args = parser.parse_args()

    return args
