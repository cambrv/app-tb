import fitz 

doc = fitz.open("file.pdf")
contenido = ""

for page in doc:
    blocks = page.get_text("dict")["blocks"]
    for block in blocks:
        if "lines" in block:
            bbox = block["bbox"]
            if bbox[1] > 50 and bbox[3] < page.rect.height - 50: 
                for line in block["lines"]:
                    line_text = " ".join([span["text"] for span in line["spans"]])
                    contenido += line_text + "\n"

with open("file.txt", "w", encoding="utf-8") as f:
    f.write(contenido)
