import os
import argparse
import glob
from collections import Counter

import numpy as np
import cv2
from PIL import Image
import random
import colorsys
random.seed(4747)

from textures_using_anchors import grid_offset_by_anchors, load_known_templates

def get_colorset(num_colors):
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    random.shuffle(colors)
    return colors

def main(args):
    if args.folder != '':
        file_paths = glob.glob(os.path.join(args.folder, '*.png'))
    else:
        file_paths = [args.file]
    
    os.makedirs('segmented_images', exist_ok=True)
    for file_path in file_paths:
        file_name = os.path.split(args.file)[1]
        name = os.path.splitext(file_name)[0]
        # Load in with opencv
        orig_image = cv2.imread(args.file, cv2.IMREAD_UNCHANGED)
        if orig_image.shape[2] == 4:
            orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGRA2BGR)

        h, w, *_ = orig_image.shape

        # Specific to SMB, most common top row pixel will be background
        top_row = []
        for i in range(w):
            pixel = orig_image[0, i, :3]
            top_row.append((pixel[0], pixel[1], pixel[2]))
        x = max(top_row, key=top_row.count)
        background_color = np.array([x[0], x[1], x[2], 255])
        
        print(f"Loading textures")
        textures = load_known_templates(args.textures, background_color)
        print(f"Num textures: {len(textures)}")

        print(f"Loading anchors")
        anchors = load_known_templates(args.anchors, background_color)
        print(f"Num anchors: {len(anchors)}")

        safe_img = np.copy(orig_image)
        
        print(f"Getting offset by anchors")
        args.x_offset, args.y_offset = grid_offset_by_anchors(safe_img, anchors, args)
        print(f"Got offset: {args.x_offset}, {args.y_offset}")
        print(f"Template matching known textures")

        repainted_image, confidence = match_known_textures(safe_img, textures, args)

        print(f"Done matching")
        cv2.imwrite(f"{name}_thr_{int(args.threshold*1000)}_repaint.png", repainted_image)
        # cv2.imwrite(f"{name}_conf_thr_{int(args.threshold*1000)}.png", confidence)
        cv2.imwrite(f"{name}_orig.png", orig_image)

def match_known_textures(image, known_textures, args):
    """Takes BGR OpenCV image and dictionary of textures from load_known_textures and uses
    template matching with args.threshold to locate textures in the image. Fills an output
    image with unique colors for each texture matched on the original. Optionally saves
    templates that matched for viewing.
    
    Arguments:
        image {ndarray} -- OpenCV Image to match textures on
        known_textures {dict} -- dictionary of textures and masks to match
        args {namespace} -- requires file, textures, threshold. optional: verbose, save_imgs
    
    Returns:
        [ndarray] -- repainted 3-channel image based on matched textures
    """    
    dupe_ctr = 0
    ctr = 0
    colorset = get_colorset(300)

    unmatched = []
    matched_textures = []
    matched_locations = []
    out_image = np.ones_like(image, np.uint8)
    if args.sqdiff:
        confidence_map = np.ones((image.shape[0], image.shape[1]), np.float)
        comparison = lambda x,y: x < y
        print('Compare using less than')
    else:
        confidence_map = np.zeros((image.shape[0], image.shape[1]), np.float)
        comparison = lambda x,y: x > y

    h, w, *_ = image.shape
    grid_size = 16
    # Need to treat 0,0 offset the same as others, else it will observe one more row and col
    xmax = w-grid_size + args.x_offset
    cols = (w-grid_size)//grid_size
    ymax = h-grid_size + args.y_offset
    rows = (h-grid_size)//grid_size
    included_indices = [(r*grid_size + args.y_offset, c*grid_size + args.x_offset) for r in range(rows)
                         for c in range(cols)]

    for file_name, data in known_textures.items():
        old_texture, mask, orig = data
        if args.sqdiff:
            results = cv2.matchTemplate(image, old_texture, cv2.TM_SQDIFF_NORMED)
            results = np.where(~np.isnan(results), results, 0)
            results = np.where(~np.isinf(results), results, 0)
            loc = np.where(results <= args.threshold)
        else:
            results = cv2.matchTemplate(image, old_texture, cv2.TM_CCORR_NORMED)
            results = np.where(~np.isnan(results), results, 0)
            results = np.where(~np.isinf(results), results, 0)
            loc = np.where(results >= args.threshold)

        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(results)
        full_matches = list(zip(*loc[::1]))

        # rows, cols = grid_using_crop(grid_offset_x=3, grid_offset_y=7)
        # print(cols)
        matches = list(filter(lambda t: t in included_indices, full_matches))

        if len(full_matches) > 9000:
            print(f"OVER 9000 {file_name}: {len(full_matches)}. minimized: {len(matches)}")
        if len(matches) != 0 and len(matches) < 9000:
            curr_color = colorset[ctr]
            ctr += 1

            matched_textures.append(file_name)

            if args.save_imgs:
                cv2.imwrite(f"temp/{file_name}.png", old_texture)

                color_mask = np.dstack((np.copy(mask), np.copy(mask), np.copy(mask)))
                color_mask[mask == 255] = [curr_color[0]*255, curr_color[1]*255, curr_color[2]*255]
                cv2.imwrite(f"temp/mask_{file_name}.png", color_mask)
                cv2.imwrite(f"temp/orig_{file_name}.png", orig)


            for match_row, match_col in matches:
                matched_locations.append((match_row, match_col))
                indices = [(r, c) for r in range(mask.shape[0])
                         for c in range(mask.shape[1])]
                new_confidence = results[match_row, match_col]
                for (r, c) in indices:
                    old_confidence = confidence_map[r, c]
                    if mask[r, c] == 255:
                        if old_confidence != 0.0:
                            dupe_ctr += 1
                        out_image[match_row+r,match_col+c, :] = [curr_color[0]*255, curr_color[1]*255, curr_color[2]*255]
                        confidence_map[match_row+r,match_col+c] = new_confidence
        else:
            unmatched.append(file_name)
    if args.verbose:
        
        print(sorted(matched_textures))
        print(len(unmatched))
        print(len(matched_locations))
        print(len(matched_textures))
        print(dupe_ctr)
    
    maxim = confidence_map.max()
    space = np.linspace(0, maxim*255, num=len(np.unique(confidence_map)), dtype=np.uint8)
    normalized = np.zeros_like(confidence_map, dtype=np.uint8)
    for i, val in enumerate(np.unique(confidence_map)):
        normalized[confidence_map == val] = space[i]
    # for (r,c) in included_indices:
    #     image[r, c, :] = [255,0,0]
    conf = cv2.applyColorMap(normalized, cv2.COLORMAP_AUTUMN)
    if args.save_imgs:
        # print(np.unique(confidence_map))
        cv2.imwrite(f"orig.png", image)
        cv2.imwrite(f"matched.png", out_image)
        cv2.imwrite(f"conf.png", conf)
    return out_image, conf

def parse_args():
    parser = argparse.ArgumentParser(description='Textrue match detection')

    parser.add_argument('--file', type=str, default='./smb/screens/5.png')
    parser.add_argument('--folder', type=str, default='')
    parser.add_argument('--textures', type=str, default='./smb/textures/')
    parser.add_argument('--anchors', type=str, default='./smb/anchors/')
    parser.add_argument('--threshold', type=float, default=0.98)
    parser.add_argument('--grid-size', type=int, default=16)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--save-imgs', action='store_true')
    parser.add_argument('--save-np', action='store_true')
    parser.add_argument('--sqdiff', action='store_true')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    print(f'file: {args.file}')
    print(args)
    main(args)