
import csv,json
# """
# 输入文件名：data.csv
# 输出文件名：output.json
# 转换完成！共处理15条记录
# 继续转换？(y/n): n
# """

def process_file(csv_file_path='data.csv', json_file_path = 'ouput.json'):

    with open(csv_file_path, 'r', encoding="utf-8") as csvfile, open(json_file_path, 'w', encoding="utf-8") as jsonfile:
        reader = csv.DictReader(csvfile)
        dict_row = [row for row in reader]
        n = dict_row.__len__()
        json_str = json.dumps(dict_row,ensure_ascii=False)
        jsonfile.write(json_str)
    
    print(f"转换完成！共处理{n}条记录")


if __name__ == '__main__':
    reset = True
    while reset:
        csv_file_path= input("输入文件名：")
        json_file_path = input("输出文件名: ")
        process_file( csv_file_path, json_file_path)

        while True:
            key = input("继续转换？(y/n):")
            if key == 'n' or key == 'N':
                reset = False
                break
            elif key == 'y' or key == 'Y':
                break
            else:
                continue