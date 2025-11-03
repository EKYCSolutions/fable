# FABLE

**FABLE** is an automated face accessory labeling tool powered by a **vision-language model (VLM)**. It scans through images, detects faces (optional), and uses a multimodal LLM to describe whether certain accessories (like glasses, masks, hats, etc.) are present.

This tool is particularly useful for generating labeled datasets from large collections of face images for training face accessories recognition models.

## 🚀 Features

- 🔍 **Automatic image scanning** — recursively gathers all images from a directory  
- 🧩 **Configurable labeling schema** — define accessories and descriptions in a YAML file  
- 🧠 **LLM-based annotation** — integrates with **LangChain** and **Ollama** for structured labeling  
- 👤 **Optional face detection** — detects individual faces with `face_recognition`  
- 🧵 **Parallel processing** — speeds up labeling with multi-threaded execution  
- 📝 **Resumable progress tracking** — continue from where you left off using the built-in tracker  
- 💾 **CSV output** — all annotations saved in `annotation.csv`

## 📦 Installation

> This tool uses **Langchain** to interacts with models running through [Ollama](https://ollama.ai/). Therefore, it requires **Ollama** to be installed and running locally.

1. Clone the repository:

```bash
git clone https://github.com/EKYCSolutions/fable.git
cd fable
```

2. (Optional) Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## ⚙️ Configuration

Create a YAML configuration file (e.g., `config.yaml`) like this:

```yaml
accessories:
  glasses: Eyewear consisting of lenses mounted in a frame resting on the nose and ears.
  hat: Any item worn on or covering the head.
  mask: Cloth or medical covering over the mouth and nose.

configurations:
  model: qwen2-vl:7b
  detect_faces: true
  image_extensions:
    - jpg
    - jpeg
    - png
```

### Configuration fields

| Key | Type | Description |
|-----|------|-------------|
| `accessories` | dict | Mapping of label names to their descriptions |
| `configurations.model` | str | Ollama model name (e.g. `gemma3:12b`, `qwen2-vl:7b`) |
| `configurations.detect_faces` | bool | Whether to run face detection before labeling |
| `configurations.image_extensions` | list[str] | File extensions to include when searching for images |

## 🧩 Usage

Run the tool from the command line:

```bash
python label.py /path/to/images -c config.yaml -o output_dir
```

### Arguments

| Argument | Description | Default |
|-----------|--------------|----------|
| `data_dir` | Path to image directory | — |
| `-c`, `--config` | Path to YAML configuration file | `config.yaml` |
| `-o`, `--output_dir` | Directory to store results and tracker files | `out/` |
| `-w`, `--workers` | Number of concurrent workers (threads) | `4` |
| `-v`, `--verbose` | Enable verbose logging | `False` |

## 🧠 Example Workflow

```bash
# Step 1: Prepare your dataset
mkdir data
cp /path/to/faces/*.jpg data/

# Step 2: Write config.yaml
nano config.yaml  # or use the example above

# Step 3: Start labeling
python label.py data -c config.yaml -o results -w 8
```

Progress is displayed in a live progress bar using **tqdm**:

```
Processing items:  72%|███████▏  | 72/100 [03:12<01:14,  2.37s/it]
```

If you stop and rerun, it will **resume automatically** without reprocessing finished images.

## 🧾 Output

Each processed face is saved to a CSV file (default: `out/annotation.csv`) with the following structure:

| filename | person_id | xmin | ymin | xmax | ymax | glasses | mask | hat | ... |
|-----------|------------|------|------|------|------|----------|------|------|-----|
| `images/img1.jpg` | `0` | `32` | `48` | `128` | `144` | `1` | `0` | `0` | ... |

## 🔍 How It Works

1. **Loads configuration** → from YAML  
2. **Scans image directory** → finds all images by extension  
3. **Detects faces (optional)** → via `face_recognition`  
4. **Converts each face to base64** → for LLM input  
5. **Prompts the model** → using LangChain and your accessories list  
6. **Parses structured output** → into a Pydantic model  
7. **Appends results** → to a CSV file  

## ⚡ Tips for Better Performance

- Use a **smaller model** (e.g., `qwen2-vl:2b`) for faster annotation.
- Increase `-w` (workers) for better throughput on multicore machines.
- Disable face detection (`detect_faces: false`) if all images are already cropped faces.

## 🧑‍💻 Example Output (Simplified)

```csv
filename,person_id,xmin,ymin,xmax,ymax,glasses,mask,hat
data/image1.jpg,0,12,40,120,150,1,0,0
data/image2.jpg,0,10,30,130,140,0,1,0
```

## 🧱 Project Structure

```
fable/
│
├── label.py               # Main labeling script
├── utils.py               # Helper functions
├── tracker.py             # Progress tracking
├── requirements.txt
└── configs/               # Configs  directory.
    └── config.yaml
```

## 🪪 License

Distributed under the MIT License. See [LICENSE](./LICENSE) for more information.