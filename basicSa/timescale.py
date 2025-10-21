import os

def timescalefile(inputpath, timescale, outputpath):
    with open(inputpath, 'r') as infile, open(outputpath, 'w') as outfile:
        lines = infile.readlines()

        # 遍历每一行，并根据 timescale 进行过滤
        for i, line in enumerate(lines):
            if i <= timescale:
                outfile.write(line)
            else:
                break


def timescaledir(inputpath, timescale, outputpath):
    # 遍历目录下的所有文件
    for filename in os.listdir(inputpath):
        # 检查文件是否以 .txt 结尾
        if filename.endswith('.txt'):
            print(filename)
            input_file_path = os.path.join(inputpath, filename)
            output_file_path = os.path.join(outputpath, filename)

            # 处理每个文件
            timescalefile(input_file_path, timescale, output_file_path)



# 示例调用
# satpath = "/home/yfh/Desktop/Data/sats"
# timescale = 400
#
# testpath = "/home/yfh/Desktop/Data/test"
# timescaledir(satpath, timescale, testpath)
