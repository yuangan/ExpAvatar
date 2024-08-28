import argparse
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch as th
import torch.nn.functional as F
import torchvision.transforms as transforms
from glob import glob

from utils.script_util2 import (
    model_and_diffusion_defaults,
    create_model_and_diffusioninsta_conv,
    add_dict_to_argparser,
    args_to_dict,
)

import lmdb
from io import BytesIO
from torchvision.utils import save_image

from decalib.datasets import datasets as deca_dataset

import pickle
from PIL import Image

from torchvision.transforms import ToPILImage
to_pil = ToPILImage()

# Define the desired size for interpolation
desired_size = (256, 256)
# Apply transformations to the image
transform_predrgb = transforms.Compose([
    transforms.Resize(desired_size),
    transforms.ToTensor()
])

def save_physic_cond_to_img(rendered, normal, albedo):

    # 假设你的张量名分别为 rendered, normal, albedo
    # 归一化张量值到 [0, 1] 区间
    rendered_normalized = (rendered - rendered.min()) / (rendered.max() - rendered.min())
    normal_normalized = (normal - normal.min()) / (normal.max() - normal.min())
    albedo_normalized = (albedo - albedo.min()) / (albedo.max() - albedo.min())

    # 在宽度方向上拼接三个张量
    concatenated = th.cat((rendered_normalized, normal_normalized, albedo_normalized), dim=2)

    # 将拼接后的张量转换为 PIL 图像并保存
    concatenated = concatenated.squeeze(0)  # 移除批次维度
    concatenated = concatenated.cpu().detach().numpy()  # 转换为 NumPy 数组
    concatenated = (concatenated * 255).astype('uint8')  # 转换为 [0, 255] 范围的 uint8 类型
    concatenated = concatenated.transpose(1, 2, 0)  # 将通道维度移动到最后

    img = Image.fromarray(concatenated)
    img.save('concatenated_image.png')

def read_lmdb(path, length):
    env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    normals, albedos, rendereds, instas = [], [], [], []
    transform = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    transform = transforms.Compose(transform)

    for index in range(length):
        with env.begin(write=False) as txn:
            key = f"normal_{str(index).zfill(6)}".encode("utf-8")
            normal_bytes = txn.get(key)

            key = f"albedo_{str(index).zfill(6)}".encode("utf-8")
            albedo_bytes = txn.get(key)

            key = f"rendered_{str(index).zfill(6)}".encode("utf-8")
            rendered_bytes = txn.get(key)
            
            key = f"insta_{str(index).zfill(6)}".encode("utf-8")
            insta_bytes = txn.get(key)

        buffer = BytesIO(normal_bytes)
        normal = pickle.load(buffer)

        buffer = BytesIO(albedo_bytes)
        albedo = pickle.load(buffer)

        buffer = BytesIO(rendered_bytes)
        rendered = pickle.load(buffer)
        
        buffer = BytesIO(insta_bytes)
        insta = Image.open(buffer)
        insta = transform(insta)

        normals.append(normal)
        albedos.append(albedo)
        rendereds.append(rendered)
        instas.append(insta)

    return {
            "normals": normals,
            "albedos": albedos,
            "rendereds": rendereds,
            "instas": instas,
        }

def create_inter_data(dataset, modes, physics_conds, idx, meanshape_path=""):
    meanshape = None
    if os.path.exists(meanshape_path):
        print("use meanshape: ", meanshape_path)
        with open(meanshape_path, "rb") as f:
            meanshape = pickle.load(f)
    # else:
        # print("not use meanshape")

    image2 = dataset[-1]["original_image"].unsqueeze(0).to("cuda")
    for i in range(len(dataset) - 1):

        # To align the face when the pose is changing
        original_image = dataset[i]["original_image"].unsqueeze(0).to("cuda")

        for mode in modes:
            batch = {}
            batch["image"] = original_image * 2 - 1
            batch["image2"] = image2 * 2 - 1
            batch["mode"] = mode
            batch["rendered"] = physics_conds['rendereds'][i].cuda().unsqueeze(0)
            batch["normal"] = physics_conds['normals'][i].cuda().unsqueeze(0)
            batch["albedo"] = physics_conds['albedos'][i].cuda().unsqueeze(0)
            batch["insta"] = physics_conds['instas'][i].cuda().unsqueeze(0)
            yield batch


def main():
    args = create_argparser().parse_args()

    length = args.length
    subjectname=args.subjectname
    physics_conds = read_lmdb(f'./data_lmdb/{subjectname}_insta_wobody_test_conf_{length}.lmdb', length)

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusioninsta_conv(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    ckpt = th.load(args.model_path)

    model.load_state_dict(ckpt)
    model.to("cuda")
    model.eval()

    imagepath_list = []

    first_frame = f'../stepI/data/{subjectname}/images_wobody/00000.png'
    sources = [first_frame]
    

    for idx, sou in (enumerate(sources)):
        # breakpoint()
        args.source = first_frame
        args.target = first_frame
        imagepath_list = []
        if os.path.isdir(args.source):
            imagepath_list += (
                glob(args.source + "/*.jpg")
                + glob(args.source + "/*.png")
                + glob(args.source + "/*.bmp")
            )
        else:
            imagepath_list += [args.source]*length
        imagepath_list += [args.target]
        sou_name = os.path.basename(args.source)

        dataset = deca_dataset.TestData(imagepath_list, iscrop=True, size=args.image_size)

        modes = args.modes.split(",")

        data = create_inter_data(dataset, modes, physics_conds, idx, args.meanshape)

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        os.system("mkdir -p " + args.output_dir)
        
        # torch.Size([1, 3, 256, 256])
        noise = th.randn(1, 3, args.image_size, args.image_size).to("cuda")

        vis_dir = args.output_dir

        idx = 0
        t = th.tensor([5], dtype=th.long).cuda()
        for index_test, batch in tqdm(enumerate(data)):
            # pred_path = f'/data1/gy/INSTA-pytorch/workspace/yufeng_pfv_uncond_grid_level4/results/images/ep0070_test/rgb/{index_test+1}.png'
            # pred_rgb = Image.open(pred_path)
            # pred_rgb = transform_predrgb(pred_rgb).unsqueeze(0)
            # latents = F.interpolate(pred_rgb, (256, 256), mode='bilinear', align_corners=False) * 2 - 1

            image = batch["image"]
            image2 = batch["image2"]
            rendered, normal, albedo, insta = batch["rendered"], batch["normal"], batch["albedo"], batch["insta"]

            # noise = th.randn_like(insta)
            latents_noisy = diffusion.q_sample(insta.cuda(), t, noise.cuda())
            
            # if idx%20==0:
            #     save_physic_cond_to_img(rendered, normal, albedo)
            #     breakpoint()
            # save_physic_cond_to_img(rendered, normal, albedo)
            
            physic_cond = th.cat([rendered, normal, albedo], dim=1)
            image = image
            physic_cond = physic_cond
            with th.no_grad():
                if batch["mode"] == "latent":
                    detail_cond = model.encode_cond(image2)
                else:
                    detail_cond = model.encode_cond(image)

            sample = sample_fn(
                model,
                (1, 3, args.image_size, args.image_size),
                # noise=noise,
                noise=latents_noisy,
                clip_denoised=args.clip_denoised,
                model_kwargs={"physic_cond": physic_cond, "detail_cond": detail_cond, "insta": insta},
                start_t=t.cpu().numpy()[0]+1,
            )
            sample = (sample + 1) / 2.0
            sample = sample.contiguous()
            # save_image(sample, f'tmp/latents_noisy_step_{idx}_{t.cpu().numpy()[0]}.png')
            # save_image(latents_noisy, f'tmp/noisy_{idx}_{t.cpu().numpy()[0]}.png')

            save_image(
                sample, os.path.join(vis_dir, f'{idx+1}.png')
            )
            idx += 1
        os.makedirs(f'{vis_dir}/rgb', exist_ok=True)
        os.system(f'mv {vis_dir}/*png {vis_dir}/rgb/')

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        use_ddim=True,
        model_path="",
        source="",
        target="",
        output_dir="",
        modes="pose,exp,light",
        meanshape="",
        subjectname="",
        length=0
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
