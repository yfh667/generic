import originpro as op

def ensure_sheet(book_name: str = 'Book1',
                 sheet_name: str = 'Ration',
                 *,
                 clear_existing: bool = False,
                 activate: bool = True):
    """
    在指定工作簿中获取/创建工作表。
    - 若表已存在：返回该表；可选清空内容。
    - 若表不存在：创建并返回。
    - 可选设为激活层，便于后续写入。

    返回：originpro 的 Worksheet 对象
    """
    # 1) 找到工作簿（type='w'）
    bk = op.find_book('w', book_name)
    if bk is None:
        # 若确实要强制创建，可改为：bk = op.new_book('w', lname=book_name)
        raise RuntimeError(f"未找到工作簿 '{book_name}'。请确认已在当前工程中。")

    # 2) 查找是否已有同名工作表（优先 lname，其次短名）
    target = None
    for w in bk:
        lname = getattr(w, 'lname', None) or w.name
        if lname == sheet_name:
            target = w
            break

    # 3) 创建或复用
    if target is None:
        target = bk.add_sheet(sheet_name, active=True)
        #target = op.new_sheet('w', lname=sheet_name, book=bk)
    elif clear_existing:
        try:
            target.clear()
        except Exception:
            # 某些版本无 clear() 可直接覆盖 from_df
            pass

    # 4) 设为激活层（可选）
    if activate:
        try:
            target.activate()
        except Exception:
            pass

    return target


#     ws = write2origin.ensure_sheet('Book1', 'Ration', clear_existing=True, activate=True)
#     ws.from_df(df_sum[cols])