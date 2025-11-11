import xml.etree.ElementTree as ET
import re
from xml.dom import minidom
import os
import ast

import genaric2.tegnode as tegnode
# ========= super fast XML -> nodes (单文件) =========
try:
    # lxml 更快，且 deep-clear/huge_tree 可进一步省内存
    from lxml import etree as _ET
    _HAS_LXML = True
except Exception:
    import xml.etree.ElementTree as _ET
    _HAS_LXML = False


print(_HAS_LXML)
from xml.sax.saxutils import quoteattr
from typing import Dict, Tuple


def nodes_to_xml(nodes: Dict[Tuple[int,int,int], object], filename: str):
    """
    Streaming writer: very fast & low memory.
    nodes: dict[(x, y, step)] -> node_obj (has attributes used below)
    """
    with open(filename, 'w', encoding='utf-8', newline='') as f:
        write = f.write
        qa = quoteattr  # local binding for speed

        write('<?xml version="1.0" encoding="utf-8"?>\n<Nodes>\n')
        for (x, y, step), node in nodes.items():
            # read attributes once (avoid repeated attribute lookups)
            asc_flag = getattr(node, "asc_nodes_flag", "")
            rn = getattr(node, "rightneighbor", None)
            ln = getattr(node, "leftneighbor", None)
            state = getattr(node, "state", "")
            imp   = getattr(node, "importance", "")

            write("  <Node ")
            write('coordination=' + qa(f"{x},{y},{step}"))
            write(' asc_nodes_flag=' + qa(str(asc_flag)))
            write(' rightneighbor=' + qa("None" if rn is None else str(rn)))
            write(' leftneighbor=' + qa("None"  if ln is None else str(ln)))
            write(' state=' + qa(str(state)))
            write(' importance=' + qa(str(imp)))
            write("/>\n")
        write("</Nodes>\n")

def nodes_to_xml_test(nodes: Dict[Tuple[int,int,int], object], filename: str):
    """
    Streaming writer: very fast & low memory.
    nodes: dict[(x, y, step)] -> node_obj (has attributes used below)
    """
    with open(filename, 'w', encoding='utf-8', newline='') as f:
        write = f.write
        qa = quoteattr  # local binding for speed

        write('<?xml version="1.0" encoding="utf-8"?>\n<Nodes>\n')
        for (x, y, step), node in nodes.items():
            # read attributes once (avoid repeated attribute lookups)
            asc_flag = getattr(node, "asc_nodes_flag", "")
            rn = getattr(node, "rightneighbor", None)
            ln = getattr(node, "leftneighbor", None)
            leftstate = getattr(node, "left_state", -1)
            right_state = getattr(node, "right_state", -1)

            node_type   = getattr(node, "node_type", "")
            timelast = getattr(node, "timelast", "")

            write("  <Node ")
            write('coordination=' + qa(f"{x},{y},{step}"))
            write(' asc_nodes_flag=' + qa(str(asc_flag)))
            write(' rightneighbor=' + qa("None" if rn is None else str(rn)))
            write(' leftneighbor=' + qa("None"  if ln is None else str(ln)))
            write(' left_state=' + qa(str(leftstate)))
            write(' right_state=' + qa(str(right_state)))
            write(' node_type=' + qa(str(node_type)))
            write(' timelast=' + qa(str(timelast)))
            write("/>\n")
        write("</Nodes>\n")

# def nodes_to_xml(nodes, filename):
#     """
#     将 nodes 字典保存为 XML 文件
#     nodes: dict[(x, y, step)] -> tegnode
#     filename: 保存的xml路径
#     """
#     root = ET.Element("Nodes")
#     for coord, node in nodes.items():
#         node_elem = ET.SubElement(root, "Node")
#         node_elem.set("coordination", f"{coord[0]},{coord[1]},{coord[2]}")
#         node_elem.set("asc_nodes_flag", str(node.asc_nodes_flag))
#         node_elem.set("rightneighbor", str(node.rightneighbor) if node.rightneighbor is not None else "None")
#         node_elem.set("leftneighbor", str(node.leftneighbor) if node.leftneighbor is not None else "None")
#         node_elem.set("state", str(node.state))
#         node_elem.set("importance", str(node.importance))
#
#     # 格式化输出
#     xml_str = ET.tostring(root, encoding='utf-8')
#     dom = minidom.parseString(xml_str)
#     pretty_xml = dom.toprettyxml(indent="  ")
#
#     with open(filename, 'w', encoding='utf-8') as f:
#         f.write(pretty_xml)

def _parse_tuple_int(s: str):
    """把 '1,2,3' 或 '(1, 2, 3)' 转成 tuple[int,...]；'None'/空 -> None（比 ast.literal_eval 更快）。"""
    if not s or s == 'None':
        return None
    s = s.strip()
    if s and s[0] == '(' and s[-1] == ')':
        s = s[1:-1]
    # 允许有空格
    parts = s.split(',')
    # 过滤空片段，防止 '1,2,' 之类
    return tuple(int(p.strip()) for p in parts if p.strip())


def xml_to_nodes(filename, tegnode_cls):
    """
    Fast XML -> nodes(dict) for old schema:
      attrs: coordination, asc_nodes_flag, rightneighbor, leftneighbor, state, importance
    - Prefer lxml iterparse(tag='Node', huge_tree=True) + deep-clear
    - Fallback to stdlib xml.etree.ElementTree
    - No ast.literal_eval; use lightweight tuple parser
    """
    nodes = {}
    cls = tegnode_cls
    ptuple = _parse_tuple_int
    ATTR_TRUE = {'True', 'true', '1'}

    # 1) build iterparse context (tag filter when supported)
    try:
        if _HAS_LXML:
            context = _ET.iterparse(str(filename), events=('end',), tag='Node', huge_tree=True)
        else:
            context = _ET.iterparse(str(filename), events=('end',), tag='Node')
        use_tag_filter = True
    except TypeError:
        # some backends don't support tag=/huge_tree=
        if _HAS_LXML:
            context = _ET.iterparse(str(filename), events=('end',), huge_tree=True)
        else:
            context = _ET.iterparse(str(filename), events=('end',))
        use_tag_filter = False

    try:
        for _, elem in context:
            if not use_tag_filter and elem.tag != 'Node':
                if _HAS_LXML:
                    while elem.getprevious() is not None:
                        del elem.getparent()[0]
                elem.clear()
                continue

            at = elem.attrib
            coord = at.get('coordination')
            if not coord:
                if _HAS_LXML:
                    while elem.getprevious() is not None:
                        del elem.getparent()[0]
                elem.clear()
                continue

            # coords (faster than map+tuple)
            try:
                x_str, y_str, s_str = coord.split(',')
                coords = (int(x_str), int(y_str), int(s_str))
            except Exception:
                if _HAS_LXML:
                    while elem.getprevious() is not None:
                        del elem.getparent()[0]
                elem.clear()
                continue

            # neighbors (lightweight parser, handles None / "(..)" / "a,b,c")
            rn = ptuple(at.get('rightneighbor'))
            ln = ptuple(at.get('leftneighbor'))

            # flags / ints
            asc_flag = (at.get('asc_nodes_flag') in ATTR_TRUE)
            s = at.get('state');       state = int(s) if s and s.strip() else -1
            im = at.get('importance'); importance = int(im) if im and im.strip() else 0

            node = cls(
                asc_nodes_flag=asc_flag,
                rightneighbor=rn,
                leftneighbor=ln,
                state=state,
                importance=importance
            )
            nodes[coords] = node

            # free element memory
            if _HAS_LXML:
                elem.clear()
                while elem.getprevious() is not None:
                    del elem.getparent()[0]
            else:
                elem.clear()

    except Exception as e:
        print(f"解析XML时出错: {e}")
        return {}  # 返回空 dict 更稳

    return nodes


def xml_to_nodes_test(filename, tegnode_cls):
    """
    Fast XML -> nodes(dict) for old schema:
      attrs: coordination, asc_nodes_flag, rightneighbor, leftneighbor, state, importance
    - Prefer lxml iterparse(tag='Node', huge_tree=True) + deep-clear
    - Fallback to stdlib xml.etree.ElementTree
    - No ast.literal_eval; use lightweight tuple parser
    """
    nodes = {}
    cls = tegnode_cls
    ptuple = _parse_tuple_int
    ATTR_TRUE = {'True', 'true', '1'}

    # 1) build iterparse context (tag filter when supported)
    try:
        if _HAS_LXML:
            context = _ET.iterparse(str(filename), events=('end',), tag='Node', huge_tree=True)
        else:
            context = _ET.iterparse(str(filename), events=('end',), tag='Node')
        use_tag_filter = True
    except TypeError:
        # some backends don't support tag=/huge_tree=
        if _HAS_LXML:
            context = _ET.iterparse(str(filename), events=('end',), huge_tree=True)
        else:
            context = _ET.iterparse(str(filename), events=('end',))
        use_tag_filter = False

    try:
        for _, elem in context:
            if not use_tag_filter and elem.tag != 'Node':
                if _HAS_LXML:
                    while elem.getprevious() is not None:
                        del elem.getparent()[0]
                elem.clear()
                continue

            at = elem.attrib
            coord = at.get('coordination')
            if not coord:
                if _HAS_LXML:
                    while elem.getprevious() is not None:
                        del elem.getparent()[0]
                elem.clear()
                continue

            # coords (faster than map+tuple)
            try:
                x_str, y_str, s_str = coord.split(',')
                coords = (int(x_str), int(y_str), int(s_str))
            except Exception:
                if _HAS_LXML:
                    while elem.getprevious() is not None:
                        del elem.getparent()[0]
                elem.clear()
                continue

            # neighbors (lightweight parser, handles None / "(..)" / "a,b,c")
            rn = ptuple(at.get('rightneighbor'))
            ln = ptuple(at.get('leftneighbor'))

            # flags / ints
            asc_flag = (at.get('asc_nodes_flag') in ATTR_TRUE)
            s1 = at.get('left_state');       left_state = int(s1) if s1 and s1.strip() else -1
            s2 = at.get('right_state');
            right_state = int(s2) if s2 and s2.strip() else -1
            node_type1 = at.get('node_type'); node_type = int(node_type1) if node_type1 and node_type1.strip() else 0
            timelast1 = at.get('timelast'); timelast = int(timelast1) if timelast1 and timelast1.strip() else 0


            node = cls(
                asc_nodes_region_id=asc_flag,
                rightneighbor=rn,
                leftneighbor=ln,
                left_state=left_state,
                right_state=right_state,
                node_type=node_type,

                timelast=timelast
            )
            nodes[coords] = node

            # free element memory
            if _HAS_LXML:
                elem.clear()
                while elem.getprevious() is not None:
                    del elem.getparent()[0]
            else:
                elem.clear()

    except Exception as e:
        print(f"解析XML时出错: {e}")
        return {}  # 返回空 dict 更稳

    return nodes
import gzip

def nodes_to_xml2(
    nodes: Dict[Tuple[int,int,int], object],
    filename: str,
    *,
    sort_keys: bool = False,          # 需要稳定顺序就开
    gzip_if_endswith: bool = True,    # 以 .gz 结尾时自动写 gzip
    buffer_bytes: int = 1024 * 1024   # 大缓冲，减少系统调用
):
    """
    Faster replacement of nodes_to_xml2:
      - Streaming write (no ElementTree/minidom)
      - Optional gzip if filename endswith .gz/.gzip
      - Same attributes: coordination, asc_nodes_region_id, right/leftneighbor, left/right_state
    """
    qa = quoteattr  # local binding

    use_gzip = gzip_if_endswith and str(filename).lower().endswith((".gz", ".gzip"))
    opener = (lambda p, mode: gzip.open(p, mode, encoding="utf-8", newline=""))
    if not use_gzip:
        opener = (lambda p, mode: open(p, mode, encoding="utf-8", newline="", buffering=buffer_bytes))

    with opener(filename, "wt") as f:
        write = f.write
        write('<?xml version="1.0" encoding="utf-8"?>\n<Nodes>\n')

        items = nodes.items()
        if sort_keys:
            items = sorted(items)  # 按 (x,y,step) 排序

        for (x, y, step), node in items:
            # 读取属性一次，减少 getattr 次数
            ascn = getattr(node, "asc_nodes_region_id", getattr(node, "asc_nodes_flag", -1))
            rn   = getattr(node, "rightneighbor", None)
            ln   = getattr(node, "leftneighbor", None)
            ls   = getattr(node, "left_state", -1)
            rs   = getattr(node, "right_state", -1)

            # 保持你现有的字符串格式：None 或 tuple 的 str()
            rn_str = "None" if rn is None else str(rn)
            ln_str = "None" if ln is None else str(ln)

            # 单次拼好一行再写（比多次 write 更省函数开销）
            line = (
                "  <Node "
                "coordination=" + qa(f"{x},{y},{step}") +
                " asc_nodes_region_id=" + qa(str(ascn)) +
                " rightneighbor=" + qa(rn_str) +
                " leftneighbor="  + qa(ln_str) +
                " left_state="    + qa(str(ls)) +
                " right_state="   + qa(str(rs)) +
                "/>\n"
            )
            write(line)

        write("</Nodes>\n")
# def nodes_to_xml2(nodes, filename):
#     """
#     将 nodes 字典保存为 XML 文件
#     nodes: dict[(x, y, step)] -> tegnode
#     filename: 保存的xml路径
#     """
#     root = ET.Element("Nodes")
#     for coord, node in nodes.items():
#         node_elem = ET.SubElement(root, "Node")
#         node_elem.set("coordination", f"{coord[0]},{coord[1]},{coord[2]}")
#         node_elem.set("asc_nodes_region_id", str(node.asc_nodes_region_id))
#         node_elem.set("rightneighbor", str(node.rightneighbor) if node.rightneighbor is not None else "None")
#         node_elem.set("leftneighbor", str(node.leftneighbor) if node.leftneighbor is not None else "None")
#         node_elem.set("left_state", str(node.left_state))
#         node_elem.set("right_state", str(node.right_state))
#
#     # 格式化输出
#     xml_str = ET.tostring(root, encoding='utf-8')
#     dom = minidom.parseString(xml_str)
#     pretty_xml = dom.toprettyxml(indent="  ")
#
#     with open(filename, 'w', encoding='utf-8') as f:
#         f.write(pretty_xml)






def xml_to_nodes2(filename, tegnode_cls):
    """
    快速 XML -> nodes(dict)：
      - lxml: iterparse(huge_tree=True) + deep-clear
      - 兼容 stdlib xml.etree
      - 如果某实现不支持 tag=，退化为循环内判断 elem.tag
    返回: dict[(x, y, step)] -> tegnode
    """
    nodes = {}
    cls = tegnode_cls
    ptuple = _parse_tuple_int
    ATTR_TRUE = {'True', 'true', '1'}

    # 1) 建 iterparse 上下文（兼容不同实现）
    try:
        if _HAS_LXML:
            context = _ET.iterparse(str(filename), events=('end',), tag='Node', huge_tree=True)
        else:
            context = _ET.iterparse(str(filename), events=('end',), tag='Node')
        use_tag_filter = True
    except TypeError:
        # 某些实现没有 tag= 或 huge_tree=；退化为不带 tag
        if _HAS_LXML:
            context = _ET.iterparse(str(filename), events=('end',), huge_tree=True)
        else:
            context = _ET.iterparse(str(filename), events=('end',))
        use_tag_filter = False

    try:
        for _, elem in context:
            if use_tag_filter is False and elem.tag != 'Node':
                # 手动过滤
                if _HAS_LXML:
                    # lxml deep-clear
                    while elem.getprevious() is not None:
                        del elem.getparent()[0]
                elem.clear()
                continue

            at = elem.attrib
            coord = at.get('coordination')
            if not coord:
                if _HAS_LXML:
                    while elem.getprevious() is not None:
                        del elem.getparent()[0]
                elem.clear()
                continue

            # 坐标解析（比 map+tuple 略快）
            try:
                x_str, y_str, s_str = coord.split(',')
                coords = (int(x_str), int(y_str), int(s_str))
            except Exception:
                if _HAS_LXML:
                    while elem.getprevious() is not None:
                        del elem.getparent()[0]
                elem.clear()
                continue

            rn = ptuple(at.get('rightneighbor'))
            ln = ptuple(at.get('leftneighbor'))

            asc_flag = (at.get('asc_nodes_region_id') in ATTR_TRUE)

            ls = at.get('left_state');   left_state  = int(ls) if ls and ls.strip() else -1
            rs = at.get('right_state');  right_state = int(rs) if rs and rs.strip() else -1

            node = cls(
                asc_nodes_region_id=asc_flag,
                rightneighbor=rn,
                leftneighbor=ln,
                left_state=left_state,
                right_state=right_state,
            )
            nodes[coords] = node

            # 清理：先 clear，再剥离已处理兄弟（lxml）
            if _HAS_LXML:
                elem.clear()
                while elem.getprevious() is not None:
                    del elem.getparent()[0]
            else:
                elem.clear()

    except Exception as e:
        print(f"解析XML时出错: {e}")
        return {}     # 返回空 dict，避免后续 update(None) 报错

    return nodes

#
# def xml_to_nodes2(filename, tegnode_cls):
#     """
#     更快的 XML -> nodes(dict) 读取：
#       - ET.iterparse 流式解析（边读边清理，省内存、快）
#       - 用 _parse_tuple_int() 替代 ast.literal_eval()
#     返回: dict[(x, y, step)] -> tegnode
#     """
#     nodes = {}
#     try:
#         # 只监听元素结束事件；到 </Node> 时处理该节点
#         for event, elem in ET.iterparse(filename, events=('end',)):
#             if elem.tag != 'Node':
#                 continue
#
#             at = elem.attrib  # 局部绑定，少一次查找
#             coord_str = at.get('coordination')
#             if not coord_str:   # 缺关键字段就跳过
#                 elem.clear()
#                 continue
#
#             # 坐标
#             coords = tuple(int(x.strip()) for x in coord_str.split(','))
#
#             # 其他属性（沿用你原来的字段映射/默认值）
#             asc_nodes_region_id = (at.get('asc_nodes_region_id') == 'True')
#
#             # 邻居（更轻量）
#             rightneighbor = _parse_tuple_int(at.get('rightneighbor'))
#             leftneighbor  = _parse_tuple_int(at.get('leftneighbor'))
#
#             left_state  = int(at.get('left_state',  '-1'))
#             right_state = int(at.get('right_state', '-1'))
#
#             # 构造节点对象（保持你原来的入参名）
#             node = tegnode_cls(
#                 asc_nodes_region_id=asc_nodes_region_id,
#                 rightneighbor=rightneighbor,
#                 leftneighbor=leftneighbor,
#                 left_state=left_state,
#                 right_state=right_state,
#             )
#             nodes[coords] = node
#
#             # 关键：清理已处理元素，释放内存（大文件明显加速）
#             elem.clear()
#
#     except Exception as e:
#         print(f"解析XML时出错: {e}")
#         return None
#
#     return nodes

_ATTR_TRUE = {'True', 'true', '1'}

def iter_nodes2(filename, tegnode_cls):
    """
    迭代器：逐个 yield (coords, node)
    - lxml: iterparse(huge_tree=True) + deep-clear
    - 兼容 stdlib；不支持 tag= 时退化到手动过滤
    """
    cls = tegnode_cls
    ptuple = _parse_tuple_int
    ATTR_TRUE = {'True', 'true', '1'}

    try:
        if _HAS_LXML:
            context = _ET.iterparse(str(filename), events=('end',), tag='Node', huge_tree=True)
        else:
            context = _ET.iterparse(str(filename), events=('end',), tag='Node')
        use_tag_filter = True
    except TypeError:
        if _HAS_LXML:
            context = _ET.iterparse(str(filename), events=('end',), huge_tree=True)
        else:
            context = _ET.iterparse(str(filename), events=('end',))
        use_tag_filter = False

    for _, elem in context:
        if use_tag_filter is False and elem.tag != 'Node':
            if _HAS_LXML:
                while elem.getprevious() is not None:
                    del elem.getparent()[0]
            elem.clear()
            continue

        at = elem.attrib
        coord = at.get('coordination')
        if not coord:
            if _HAS_LXML:
                while elem.getprevious() is not None:
                    del elem.getparent()[0]
            elem.clear()
            continue

        try:
            x_str, y_str, s_str = coord.split(',')
            coords = (int(x_str), int(y_str), int(s_str))
        except Exception:
            if _HAS_LXML:
                while elem.getprevious() is not None:
                    del elem.getparent()[0]
            elem.clear()
            continue

        rn = ptuple(at.get('rightneighbor'))
        ln = ptuple(at.get('leftneighbor'))

        asc_flag = (at.get('asc_nodes_region_id') in ATTR_TRUE)
        ls = at.get('left_state');   left_state  = int(ls) if ls and ls.strip() else -1
        rs = at.get('right_state');  right_state = int(rs) if rs and rs.strip() else -1

        node = cls(
            asc_nodes_region_id=asc_flag,
            rightneighbor=rn,
            leftneighbor=ln,
            left_state=left_state,
            right_state=right_state,
        )
        yield coords, node

        if _HAS_LXML:
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
        else:
            elem.clear()


def _resolve_cls_spec(tegnode_cls) -> Tuple[str, str]:
    """
    接受类对象或 'pkg.mod:Class' / 'pkg.mod.Class' 字符串，返回 (module_name, class_name)
    """
    if isinstance(tegnode_cls, str):
        s = tegnode_cls.replace(":", ".")
        parts = s.split(".")
        if len(parts) < 2:
            raise ValueError(f"Invalid class spec: {tegnode_cls}")
        module_name = ".".join(parts[:-1])
        class_name = parts[-1]
        return module_name, class_name
    # 类对象
    module_name = getattr(tegnode_cls, "__module__", None)
    class_name = getattr(tegnode_cls, "__name__", None)
    if not module_name or not class_name:
        raise ValueError("tegnode_cls must be a class or 'pkg.mod:Class' string")
    return module_name, class_name
# 2) 顺序装载（最省内存，通常已足够快）
def load_all_nodes_sequential(paths, tegnode_cls):
    total = {}
    for p in paths:
        for coords, node in iter_nodes2(p, tegnode_cls):
            total[coords] = node
    return total




import importlib

def _import_cls(module_name: str, class_name: str):
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    return cls

from typing import Tuple, Iterable




# ===== 顶层 worker：可被 ProcessPoolExecutor pickl e =====
def _worker_load_file_to_items(path: str, module_name: str, class_name: str):
    """
    子进程/线程执行：把单个文件解析成 [(coords, node), ...]
    用 list 返回，主进程再汇总；如果文件非常大，也可以改成返回 dict。
    """
    cls = _import_cls(module_name, class_name)
    items = []
    for item in iter_nodes2(path, cls):  # 复用你已有的高效迭代器
        items.append(item)
    return items
import sys
import traceback


# 3) 并行装载（需要 lxml 才有明显收益；注意磁盘带宽）
def load_all_nodes_parallel(
    paths: Iterable[str | os.PathLike],
    tegnode_cls,
    workers: int | None = None,
    backend: str = 'thread',
):
    """
    并行读取多个 XML 并合并为 dict[(x,y,step)] -> node
    backend: 'thread' | 'process'
      - 若 _HAS_LXML=True，线程模式即可（lxml 解析释放 GIL）
      - 若 _HAS_LXML=False，建议用 'process'，才能真正多核
    """
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
    import multiprocessing as mp

    # 解析类到 (module, class) 形式，保证子进程可导入
    module_name, class_name = _resolve_cls_spec(tegnode_cls)

    paths = [str(p) for p in paths]  # 统一成字符串
    if workers is None:
        # 128 核建议给到 32~64（磁盘允许的话可以更高）
        workers = min(max(32, (os.cpu_count() or 8)//2), os.cpu_count() or 8, len(paths))

    total = {}

    if backend == 'process':
        # Windows/macOS 用 spawn 更稳；Linux 默认 fork 也 OK
        ctx = mp.get_context("spawn")
        # 限制数值库线程，避免每个子进程内再开 N 线程抢 CPU
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = {
                ex.submit(_worker_load_file_to_items, p, module_name, class_name): p
                for p in paths
            }
            for fut in as_completed(futs):
                p = futs[fut]
                try:
                    items = fut.result()
                    for coords, node in items:
                        total[coords] = node
                except Exception as e:
                    print(f"[FAIL process] {p}: {e}", file=sys.stderr)
                    traceback.print_exc()

    else:
        # 线程模式：_HAS_LXML=True 时能并行吃满核；否则可能受 GIL 限制
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(_worker_load_file_to_items, p, module_name, class_name): p
                for p in paths
            }
            for fut in as_completed(futs):
                p = futs[fut]
                try:
                    items = fut.result()
                    for coords, node in items:
                        total[coords] = node
                except Exception as e:
                    print(f"[FAIL thread] {p}: {e}", file=sys.stderr)
                    traceback.print_exc()

    return total

if __name__ == '__main__':

    # 保存
    #nodes_to_xml(nodes, "test_nodes.xml")

    # 读取    dummy_file_name = "E:\\code\\paper_dataresult\\station_visible_satellites_100_test.xml"
    nodes_loaded = xml_to_nodes("test_nodes.xml", tegnode.tegnode)

     # print(nodes)
