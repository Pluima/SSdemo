import librosa
import pickle
import os
import json
import csv
from pathlib import Path

# 路径配置
METADATA_DIR = "/home/qysun/Neuro-SS/dataset/clarity/clarity_data/aad/metadata"
SCENES_BASE_DIR = "/home/qysun/Neuro-SS/dataset/clarity/clarity_data/aad"
SOURCE_DIR2 = "/home/qysun/Neuro-SS/dataset/yufei/aishell-3-data"
OUTPUT_CSV = "mix_sources_aishell.csv"
SOURCE_DIR = "/home/qysun/Neuro-SS/dataset/clarity/clarity_data/aad/targets_aishell"

# 加载房间数据
ROOM_DATA_FILE = [os.path.join(METADATA_DIR, "room.train.aad.json"),
os.path.join(METADATA_DIR, "rooms.kul.json"),
os.path.join(METADATA_DIR, "rooms.cafe.json"),
os.path.join(METADATA_DIR, "rooms.tv.json"),
os.path.join(METADATA_DIR, "rooms.office.json"),

]
room_data = {}

def load_room_data():
    """加载房间数据，建立房间名称到数据的映射"""
    global room_data
    for room_file in ROOM_DATA_FILE:
        with open(room_file, 'r') as f:
            rooms = json.load(f)
        for room in rooms:
            room_data[room['name']] = room

def calculate_spatial_vector(room_name, position_index):
    """计算source相对于麦克风的空间向量"""
    if room_name not in room_data:
        return None, None, None

    room = room_data[room_name]
    listener_pos = room['listener']['position']
    interferers = room['interferers']

    # position_index 从1开始，对应 interferers[0], interferers[1], ...
    if position_index > len(interferers):
        return None, None, None

    source_pos = interferers[position_index - 1]['position']

    # 计算空间向量: source - listener
    vector = [
        source_pos[0] - listener_pos[0],
        source_pos[1] - listener_pos[1],
        source_pos[2] - listener_pos[2]
    ]

    return vector, listener_pos, source_pos
# JSON文件到mix目录的映射
JSON_TO_MIX_DIR = {
    # "scenes.cafe25.json": "scenes_cafe_CH",
    # "scenes.tv25.json": "scenes_tv_CH",
    # "scenes.office25.json": "scenes_office_CH",
    # "scenes.kul25.json": "scenes_kul_CH",
    # "scenes.cafe1000.json": "scenes_cafe_1000",
    # "scenes.cafe1000_updated.json": "scenes_cafe_1000_updated"
    # "scenes.cafe_notsofar.json": "scenes_cafe_notsofar"
    "scenes.cafe_aishell3_concat.json": "scenes_cafe_aishell3_concat"
}

# 加载房间数据
load_room_data()

# 在alimeeting目录下递归搜索音源文件
def find_source_file(source_name):
    """在alimeeting目录下递归搜索音源文件"""
    source_path = Path(SOURCE_DIR)
    # 搜索所有可能的wav文件
    # for wav_file in source_path.rglob("*.wav"):
    #     if wav_file.stem == source_name or wav_file.name == source_name:
    #         return str(wav_file)
    source_path = Path(SOURCE_DIR2)
    for wav_file in source_path.rglob("*.wav"):
        if wav_file.stem == source_name or wav_file.name == source_name:
            return str(wav_file)
    return None

# 准备CSV数据
csv_data = []

# 处理每个JSON文件
for json_filename, mix_dir_name in JSON_TO_MIX_DIR.items():
    json_file = os.path.join(METADATA_DIR, json_filename)
    mix_dir = os.path.join(SCENES_BASE_DIR, mix_dir_name)
    
    if not os.path.exists(json_file):
        print(f"Warning: JSON file not found: {json_file}")
        continue
    
    if not os.path.exists(mix_dir):
        print(f"Warning: Mix directory not found: {mix_dir}")
        continue
    
    print(f"\nProcessing {json_filename}...")
    
    # 读取JSON文件，建立scene ID到音源名称的映射
    with open(json_file, 'r') as f:
        scenes_data = json.load(f)
    
    # 创建scene到targets的映射
    scene_to_targets = {}
    for scene in scenes_data:
        scene_id = scene['scene']
        targets = scene['targets']
        if len(targets) >= 2:
            # 按position排序，确保顺序一致
            targets_sorted = sorted(targets, key=lambda x: x['position'])
            scene_to_targets[scene_id] = [
                targets_sorted[0]['name'],
                int(targets_sorted[0]['time_start']*16000/44100),
                targets_sorted[0]['position'],
                targets_sorted[1]['name'],
                int(targets_sorted[1]['time_start']*16000/44100),
                targets_sorted[1]['position']
            ]
    
    # 找到所有mix_CH0文件
    if os.path.exists(mix_dir):
        file_lists = os.listdir(mix_dir)
        mix_files = []
        for file in file_lists:
            if "mix_CH0" in file:
                mix_files.append(os.path.join(mix_dir, file))
        
        # 处理每个mix文件
        for mix_file in sorted(mix_files):
            # 从文件名中提取scene ID (例如: S00358_mix_CH0.wav -> S00358)
            filename = os.path.basename(mix_file)
            scene_id = filename.split('_mix_CH0')[0]

            if scene_id in scene_to_targets:
                source1_name = scene_to_targets[scene_id][0]
                source1_start = scene_to_targets[scene_id][1]
                source1_position = scene_to_targets[scene_id][2]  # 添加位置信息
                source2_name = scene_to_targets[scene_id][3]
                source2_start = scene_to_targets[scene_id][4]
                source2_position = scene_to_targets[scene_id][5]  # 添加位置信息
                room_name = scene['room']  # 获取房间名称

                # 查找音源文件
                source1_path = find_source_file(source1_name)
                source2_path = find_source_file(source2_name)

                # 计算空间向量
                source1_vector, listener_pos, source1_pos = calculate_spatial_vector(room_name, source1_position)
                source2_vector, _, source2_pos = calculate_spatial_vector(room_name, source2_position)

                if source1_path and source2_path and source1_vector and source2_vector:
                    csv_data.append({
                        'mix': mix_file,
                        'source1': source1_path,
                        'source1_start': source1_start,
                        'source1_position': source1_position,
                        'source1_vector': source1_vector,
                        'source2': source2_path,
                        'source2_start': source2_start,
                        'source2_position': source2_position,
                        'source2_vector': source2_vector,
                        'listener_position': listener_pos,
                        'room': room_name
                    })
                    print(f"  Found: {scene_id} - {os.path.basename(mix_file)}")
                else:
                    print(f"  Warning: Could not find sources or spatial data for {scene_id}")
                    if not source1_path:
                        print(f"    Missing source1: {source1_name}")
                    if not source2_path:
                        print(f"    Missing source2: {source2_name}")
                    if not source1_vector:
                        print(f"    Missing spatial vector for source1 (position {source1_position})")
                    if not source2_vector:
                        print(f"    Missing spatial vector for source2 (position {source2_position})")
            else:
                print(f"  Warning: Scene {scene_id} not found in JSON file")

# 写入CSV文件
with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = [
        'mix', 'room',
        'source1', 'source1_start', 'source1_position',
        'source1_vector',
        'source2', 'source2_start', 'source2_position',
        'source2_vector',
        'listener_position',
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for row in csv_data:
        writer.writerow(row)

print(f"\nCSV file created: {OUTPUT_CSV}")
print(f"Total rows: {len(csv_data)}")