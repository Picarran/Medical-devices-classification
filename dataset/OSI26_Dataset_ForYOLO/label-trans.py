import os
import xml.etree.ElementTree as ET

def convert_voc_to_yolo(xml_folder, output_folder, class_names):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for xml_file in os.listdir(xml_folder):
        if not xml_file.endswith('.xml'):
            print(f"file : {xml_file} not an XML file")
            exit(1)

        xml_path = os.path.join(xml_folder, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 获取图像的尺寸
        image_width = int(root.find('size/width').text)
        image_height = int(root.find('size/height').text)

        # 创建对应的 .txt 文件
        txt_file_name = os.path.splitext(xml_file)[0] + '.txt'
        txt_file_path = os.path.join(output_folder, txt_file_name)

        with open(txt_file_path, 'w') as txt_file:
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                class_id = class_names.index(class_name)  # 获取类名的索引

                # 获取边界框坐标
                xmin = int(obj.find('bndbox/xmin').text)
                ymin = int(obj.find('bndbox/ymin').text)
                xmax = int(obj.find('bndbox/xmax').text)
                ymax = int(obj.find('bndbox/ymax').text)

                # 转换为 YOLO 格式的坐标（相对坐标）
                x_center = round((xmin + xmax) / 2 / image_width, 6)

                y_center = round((ymin + ymax) / 2 / image_height, 6)
                width = round((xmax - xmin) / image_width, 6)
                height = round((ymax - ymin) / image_height, 6)

                # 写入 YOLO 格式的数据
                txt_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    classes_file_path = os.path.join(output_folder, 'classes.txt')
    with open(classes_file_path, 'w') as classes_file:
        for class_name in class_names:
            classes_file.write(f"{class_name}\n")

    print(f"Conversion completed. YOLO annotations saved to {output_folder}")




xml_folder = "xmllabels/val"
output_folder = "yololabels/val"

num_classes = 26
class_names = [] 
for i in range(0, num_classes+1):
    class_names.append(str(i))

convert_voc_to_yolo(xml_folder, output_folder, class_names)
