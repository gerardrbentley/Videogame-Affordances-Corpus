import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='removes first dupe matches from line by line lists of dupe matches')

    parser.add_argument('--file', type=str, default='./dupes.text',
                        help='file path') 
    parser.add_argument('--dry', action='store_true',
                        help='dont rm files directly')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    filepath = args.file
    total_ctr = 0
    ctr = 0
    with open(filepath) as fp:
        line = fp.readline()
        while line:
            filenames = line.split('.png')
            print(f'file to keep: {filenames[0]}')
            total_ctr += len(filenames)
            last_file_idx = len(filenames)-1
            files_to_delete = list(map(lambda x: (x.strip() + ".png"), filenames[1:last_file_idx]))

            print(f'FILES to Delete {files_to_delete}')
            for x in files_to_delete:
                if os.path.exists(x) and not args.dry:
                    os.remove(x)
                    ctr += 1
                elif os.path.exists(x):
                    ctr += 1
                    print(f'dry delete: {x}')
                else:
                    ctr += 1
                    print(f"The file does not exist: {x}") 
            print('----------------------------------')
            line = fp.readline()
    print(f"Deleted {ctr} dupes from {total_ctr} with dupes")