import os
import argparse
import glob

import dask
import numpy as np
import cv2
from PIL import Image

def main(args):
    if args.folder != '':
        file_paths = glob.glob(os.path.join(args.folder, '*.png'))
    else:
        file_paths = [args.file]

    print(f"Loading textures")
    textures = load_known_templates(args.textures)
    print(f"Num textures: {len(textures)}")

    unique_sets = []
    for file_path in file_paths:
        file_name = os.path.split(file_path)[1]
        name = os.path.splitext(file_name)[0]
        # Load in with opencv
        orig_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if orig_image.shape[2] == 4:
            orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGRA2BGR)

        h, w, *_ = orig_image.shape
        safe_img = np.copy(orig_image)[args.crop_t: h
                            - args.crop_b, args.crop_l: w - args.crop_r, :]
        
        x_offset, y_offset = grid_offset_by_anchors(safe_img, textures, args)
        print(file_name)
        print(x_offset, y_offset)
        kwargs = {
            'grid_size': args.grid_size,
            'y_offset': y_offset,
            'x_offset': x_offset,
            'crop_t': args.crop_t,
            'crop_b': args.crop_b,
            'crop_l': args.crop_l,
            'crop_r': args.crop_r,
        }
        unique_sets.append(unique_tiles_using_offset(safe_img, **kwargs))
    
    all_sets = dask.compute(unique_sets)[0]
    print(f"num files: {len(file_paths)}, num sets: {len(all_sets)}")
    print(type(all_sets))
    full_set = []
    add = 0
    for i, tile_set in enumerate(all_sets):
        prev_len = len(full_set)
        print(f"Tile set num {i}, num tiles: {len(tile_set)}, prev full_set: {prev_len}, add + {add}")
        for curr_tile in tile_set:
            flag = False
            for old_tile in full_set:
                if np.array_equal(old_tile, curr_tile):
                    flag = True
                    break     
            if not flag:
                full_set.append(curr_tile)
        add = len(full_set) - prev_len
    os.makedirs('unique_textures', exist_ok=True)
    for i, tile in enumerate(full_set):
        newfile = f'unique_textures/{i:03d}.png'
        cv2.imwrite(newfile, tile)
    return

def load_known_templates(dir='./textures', fill_color=[0,0,0,255]):
    """Loads textures from pngs in directory. For textures with empty alpha areas, crops to smallest size without losing pixels and fills with fill_color.
    Yields dictionary mapping filenames to filled_texture, binary alpha mask, and original texture in 4 channel
    
    Keyword Arguments:
        dir {str} -- Where png textures are stored (default: {'./textures'})
        fill_color {list} -- Fill color for alpha channel (default: {[0,0,0,255]})
    
    Returns:
        dict[str] -> (filled_img_bgr, binary_mask, orig_img_bgra)
    """    
    known_templates = {}

    total = 0
    alpha_ct = 0
    for file in glob.glob(os.path.join(dir, '*.png')):
        total += 1
        file_name = os.path.split(file)[1]
        name = os.path.splitext(file_name)[0]
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        channels = cv2.split(img)
  

        if len(channels) == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        all_alpha = False
        if len(channels) == 4 :
            if np.count_nonzero(img[:,:,3]) == 0:
                all_alpha = True
                alpha_ct += 1
                continue
            #coords of nonzero alpha values
            y,x = img[:,:,3].nonzero()
            crop_y1 = np.min(y)
            crop_x1 = np.min(x)
            crop_x2 = np.max(x)
            crop_y2 = np.max(y)

            # cropped_tex = np.copy(img)[crop_y1:crop_y2+1, crop_x1:crop_x2+1]
            
            # channels = cv2.split(cropped_tex)
            
            # Used for painting image only where template has data
            mask = np.array(channels[3])
            mask[channels[3] == 0] = 0
            mask[channels[3] == 255] = 255

            # Add background color to template match with mask but punish background color
            out_image = np.copy(img)
            out_image[channels[3] == 0] = fill_color
            out_image = cv2.cvtColor(out_image, cv2.COLOR_BGRA2BGR)
        else:
            out_image = np.copy(img)
            mask = np.full_like(img[:,:,0], 255)

        flag = False
        for key, data in known_templates.items():
            old_out = data[0]
            if np.array_equal(old_out, out_image):
                flag = True
                break
            
        if not flag and not all_alpha:
            known_templates[name] = (out_image, mask, img)
    print(f"Total texs loaded: {total}. Minimized: {len(known_templates)}, all alpha: {alpha_ct}")
    return known_templates

def grid_offset_by_anchors(image, known_textures, args):
    x_offset_votes = np.zeros(args.grid_size)
    y_offset_votes = np.zeros(args.grid_size)
    for tex_file, data in known_textures.items():
        if tex_file[:6] == 'anchor':
            to_match = data[0]

            if args.sqdiff:
                results = cv2.matchTemplate(image, to_match, cv2.TM_SQDIFF_NORMED)
                results = np.where(~np.isnan(results), results, 0)
                results = np.where(~np.isinf(results), results, 0)
                loc = np.where(results <= args.threshold)
            else:
                results = cv2.matchTemplate(image, to_match, cv2.TM_CCORR_NORMED)
                results = np.where(~np.isnan(results), results, 0)
                results = np.where(~np.isinf(results), results, 0)
                loc = np.where(results >= args.threshold)
            
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(results)
            full_matches = list(zip(*loc[::1]))
            for match_row, match_col in full_matches:
                modrow = match_row % args.grid_size
                modcol = match_col % args.grid_size
                y_offset_votes[modrow] += 1
                x_offset_votes[modcol] += 1

    if x_offset_votes.sum() == 0 or y_offset_votes.sum() == 0:
        print('NO ANCHOR MATCH')
        return (0,0)
    else:
        if args.verbose:
            print(f"x votes: {x_offset_votes}, argmax: {np.argmax(x_offset_votes)}")
            print(f"y votes: {y_offset_votes}, argmax: {np.argmax(y_offset_votes)}")
        return(np.argmax(x_offset_votes), np.argmax(y_offset_votes))
@dask.delayed
def unique_tiles_using_offset(image, grid_size=16, y_offset=0, x_offset=0, crop_t=0, crop_b=0, crop_r=0, crop_l=0):
    h, w, *_ = image.shape
    
    # Need to treat 0,0 offset the same as others, else it will observe one more row and col
    xmax = w-grid_size + x_offset
    cols = (w-grid_size)//grid_size
    ymax = h-grid_size + y_offset
    rows = (h-grid_size)//grid_size
    included_indices = [(r*grid_size + y_offset, c*grid_size + x_offset) for r in range(rows)
                         for c in range(cols)]
    new_textures = []
    for (r,c) in included_indices:
        curr_tile = np.copy(image)[
                            r:r+grid_size, c:c+grid_size]
        
        flag = False
        for image_tile in new_textures:
            if np.array_equal(image_tile, curr_tile):
                flag = True
                break     
        if not flag:
            new_textures.append(curr_tile)
    # print(f"num new textrues: {len(new_textures)}")
    return new_textures
    
def parse_args():
    parser = argparse.ArgumentParser(description='Textrue match detection')

    parser.add_argument('--file', type=str, default='./smb/screenshots/5.png')
    parser.add_argument('--folder', type=str, default='')
    parser.add_argument('--textures', type=str, default='./smb/anchors/')
    parser.add_argument('--threshold', type=float, default=0.98)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--sqdiff', action='store_true')
    parser.add_argument('--grid-size', type=int,
                        default=16, help='grid square size')
    parser.add_argument('--crop-l', type=int,
                        default=0, help='ignore this range')
    parser.add_argument('--crop-r', type=int,
                        default=0, help='ignore this range')
    parser.add_argument('--crop-t', type=int,
                        default=0, help='ignore this range')
    parser.add_argument('--crop-b', type=int,
                        default=0, help='ignore this range')


    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    print(f'file: {args.file}')
    print(args)
    main(args)