import os.path as osp
import glob
import json
import pandas as pd


EXP_DIR = "/ubc/cs/research/kmyi/matthew/projects/ed-nerf/outputs"


def load_json(json_f):
    with open(json_f, "r") as f:
        return json.load(f)



def get_exp_dir(method_dir):
    if osp.exists(osp.join(method_dir, "eval_mean.json")):
        return method_dir
    else:
        method_dir = sorted(glob.glob(osp.join(method_dir, "*")))[-1]
        return get_exp_dir(method_dir)


def get_quant(method_dir):
    """"""
    exp_dir = get_exp_dir(method_dir)
    json_f = osp.join(exp_dir, "eval_mean.json")
    metrics = load_json(json_f)
    
    del metrics["fps"], metrics["num_rays_per_sec"]
    metrics = {k : round(v, 4) for (k,v) in metrics.items()}

    with open(osp.join(exp_dir, "commit_hash.txt"), "r") as f:
        commit_hash = f.readline()
    
    metrics["commit_hash"] = commit_hash

    return metrics

def format_to_df(input_dict):
    # Convert input dictionary to a format suitable for DataFrame construction
    df_data = {}
    for exp, metrics in input_dict.items():
        for metric, value in metrics.items():
            if metric not in df_data:
                df_data[metric] = {}
            df_data[metric][exp] = value

    # Create DataFrame
    df = pd.DataFrame.from_dict(df_data, orient='index')

    
    return df

def make_scene_table():
    """
    takes a scene and a list of experiment names
    """
    scene = "boardroom_b2_v1"
    exps = {  
        "clear_gamma" : "rgb_only",
        "clear_gamma_log_loss_evone_True" : "rgb + ev[log]",
        "clear_gamma_log_loss_identity_evone_gt_evMap-powpow_co_map" : "rgb + ev[log] + ev_mapper[powpow]"
    }

    quant_dict = {}
    scene_dir = osp.join(EXP_DIR, scene)
    for exp, header in exps.items():
        try:
            exp_dir = osp.join(scene_dir, exp)
            quant_dict[header] = get_quant(exp_dir)
        except Exception as e:
            print("error with:", header)

        
        # exp_dir = osp.join(scene_dir, exp)
        # quant_dict[header] = get_quant(exp_dir)
        

    
    df = format_to_df(quant_dict)
    
    # df.to_excel("scene_table.xlsx")
    df.to_csv("scene_table.csv")


def make_exp_table():
    """
    takes a list of experiments
    """
    scene_name = "blurry checker"
    # exp_dirs = [
    #     "/ubc/cs/research/kmyi/matthew/projects/ed-nerf/archived_outputs/outputs_0/black_seoul_b3_v3/rgb_clear_gamma_fp32",
    #     "/ubc/cs/research/kmyi/matthew/projects/ed-nerf/outputs/black_seoul_b3_v3/clear_gamma"
    # ]
    exp_dirs = ["/ubc/cs/research/kmyi/matthew/projects/ed-nerf/outputs/black_seoul_b3_v3/clear_gamma",
                "/ubc/cs/research/kmyi/matthew/projects/ed-nerf/outputs/black_seoul_b3_v3/clear_gamma_log_loss_identity_evone_gt_evMap-powpow_co_map",
                "/ubc/cs/research/kmyi/matthew/projects/ed-nerf/outputs/black_seoul_b3_v3/clear_gamma_log_loss_identity_evone_gt_evMap-powpow_co_map_evs-SO3xR3",
                "/ubc/cs/research/kmyi/matthew/projects/ed-nerf/outputs/black_seoul_b3_v3/clear_gamma_log_loss_identity_evone_gt_evMap-powpow_co_map_RGB-SO3xR3",
                "/ubc/cs/research/kmyi/matthew/projects/ed-nerf/outputs/black_seoul_b3_v3/clear_gamma_log_loss_identity_evone_gt_evMap-powpow_co_map_SO3xR3"
                ]
    headers = ["rgb_only","powpow" , "powpow + cam_opt[evs]", "powpow + cam_opt[rgb]", "powpow + cam_opt[rgb & evs]"]

    title_txt = f"# {scene_name} \n\n"
    quant_dict = {}
    for exp_dir, header in zip(exp_dirs, headers):
        assert osp.exists(exp_dir), f"{osp.basename(exp_dir)} does not exist!"
        quant_dict[header] = get_quant(exp_dir)
    
    df = format_to_df(quant_dict)
    df.to_csv("quant_table.csv")

    # mkdown_data = title_txt + df.to_markdown()

    # with open("quant_table.md", "w") as f:
    #     f.write(mkdown_data)


if __name__ == "__main__":
    make_exp_table()
    # make_scene_table()
