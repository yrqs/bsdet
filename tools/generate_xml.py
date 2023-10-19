#!/usr/bin/env python
# coding:utf-8
#from xml.etree.ElementTree import Element, SubElement, tostring
from lxml.etree import Element, SubElement, tostring
import pprint
from xml.dom.minidom import parseString

def generate_xml(xml_path, filename, width, height, depth, boxes, class_names):
    node_root = Element('annotation')

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = filename

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(width)
    node_height = SubElement(node_size, 'height')
    node_height.text = str(height)
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = str(depth)

    for box, class_name in zip(boxes, class_names):
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = class_name
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(int(box[0]))
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(int(box[1]))
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(int(box[2]))
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(int(box[3]))

    xml = tostring(node_root, pretty_print=True)  #格式化显示，该换行的换行
    dom = parseString(xml)
    with open(xml_path, 'wb+') as f:
        f.write(xml)
