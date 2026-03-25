from docx import Document

src = r"c:\Users\31876\Desktop\一种基于多视图拓扑与掩码图神经网络的风机风速缺失数据同步推演方法.docx"
out = "patent_docx_extract.txt"

doc = Document(src)
lines = []
idx = 1
for para in doc.paragraphs:
    text = para.text.strip()
    if text:
        lines.append(f"{idx:04d} {text}")
        idx += 1

with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"paras={len(lines)} out={out}")
