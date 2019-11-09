import os
import glob
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='add per game meta data to screenshot json files')

    parser.add_argument('--screenshots_dir', type=str, default='games/loz/screenshots',
                        help='path to tiles folder')
    parser.add_argument('--crop_l', type=int,
                        default=0, help='ignore area of screenshots')
    parser.add_argument('--crop_r', type=int,
                        default=0, help='ignore area of screenshots')
    parser.add_argument('--crop_t', type=int,
                        default=0, help='ignore area of screenshots')
    parser.add_argument('--crop_b', type=int,
                        default=0, help='ignore area of screenshots')
    parser.add_argument('--ui_x', type=int,
                        default=0, help='ignore area of screenshots')
    parser.add_argument('--ui_y', type=int,
                        default=0, help='ignore area of screenshots')
    parser.add_argument('--ui_width', type=int,
                        default=0, help='ignore area of screenshots')
    parser.add_argument('--ui_height', type=int,
                        default=0, help='ignore area of screenshots')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    screenshots_dir = args.screenshots_dir
    image_folders = next(os.walk(screenshots_dir))[1]

    for screenshot_uuid in image_folders:
        screenshot_file = os.path.join(screenshots_dir, screenshot_uuid, f'{screenshot_uuid}.json')

        with open(screenshot_file, mode='r') as meta_file:
            data = json.load(meta_file)
        with open(screenshot_file, mode='w') as writer:
            data['crop_l'] = args.crop_l
            data['crop_r'] = args.crop_r
            data['crop_b'] = args.crop_b
            data['crop_t'] = args.crop_t
            data['ui_x'] = args.ui_x
            data['ui_y'] = args.ui_y
            data['ui_width'] = args.ui_width
            data['ui_height'] = args.ui_height
            json.dump(data, writer)
        with open(screenshot_file, mode='r') as meta_file:
            data = json.load(meta_file)
            print(data)

    print('done')
