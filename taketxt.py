def extract_lines(input_file, output_file, interval):
    with open(input_file, 'r') as input_file:
        lines = input_file.readlines()

    extracted_lines = lines[::interval]

    with open(output_file, 'w') as output_file:
        output_file.writelines(extracted_lines)

if __name__ == "__main__":
    input_file_path = "./data.txt"
    output_file_path = "./newdata.txt"
    interval = int(input("请输入提取间隔数 n："))

    extract_lines(input_file_path, output_file_path, interval)

    print(f"提取完成，已将结果保存至 {output_file_path}")
