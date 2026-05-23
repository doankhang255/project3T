from pathlib import Path
from urllib.request import urlretrieve
import py_vncorenlp

save_dir = Path(r"C:\Users\doank\Documents\dev\project3T\News\vncorenlp")

files = {
    "VnCoreNLP-1.2.jar": "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.2.jar",

    "models/wordsegmenter/vi-vocab": "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab",
    "models/wordsegmenter/wordsegmenter.rdr": "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr",

    "models/postagger/vi-tagger": "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/postagger/vi-tagger",

    "models/ner/vi-500brownclusters.xz": "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/ner/vi-500brownclusters.xz",
    "models/ner/vi-ner.xz": "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/ner/vi-ner.xz",
    "models/ner/vi-pretrainedembeddings.xz": "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/ner/vi-pretrainedembeddings.xz",

    "models/dep/vi-dep.xz": "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/dep/vi-dep.xz",
}

save_dir.mkdir(parents=True, exist_ok=True)

for relative_path, url in files.items():
    output_path = save_dir / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"Đã có: {output_path}")
        continue

    print(f"Đang tải: {relative_path}")
    urlretrieve(url, output_path)

print("Tải xong VnCoreNLP.")

model = py_vncorenlp.VnCoreNLP(
    annotators=["wseg"],
    save_dir=str(save_dir)
)

text = "Tôi đang học xử lý ngôn ngữ tự nhiên."
print(model.word_segment(text))