import json
import os
import shutil
import glob
import random

print("=" * 80)
print("FINAL MASTER DATASET BUILDER - Aiming for 85%+ Accuracy")
print("=" * 80)

# --- 1. DEFINE OUR 6 MASTER CLASSES ---
MASTER_CLASSES = [
    "pid",
    "soiling",
    "hotspot",
    "crack",
    "shading",
    "no_anomaly"
]

# Map the dataset's class names to our new folder names
CLASS_MAP = {
    "Cracking": "crack",
    "Hot-Spot": "hotspot",
    "Hot-Spot-Multi": "hotspot",
    "Shadowing": "shading",
    "Diode": "pid",
    "Diode-Multi": "pid",
    "Soiling": "soiling",
    "No-Anomaly": "no_anomaly"
    # We ignore other classes like 'Cell', 'Offline-Module', etc.
}

# --- 2. DEFINE PATHS ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SOURCES_DIR = os.path.join(BASE_DIR, 'new_dataset_sources')
DEST_DIR = os.path.join(BASE_DIR, 'master_dataset')

print(f"\nProject Base: {BASE_DIR}")
print(f"Sources: {SOURCES_DIR}")
print(f"Destination: {DEST_DIR}\n")

# --- Ensure destination is clean before starting ---
print(f"Cleaning destination folder: {DEST_DIR}...")
if os.path.exists(DEST_DIR):
    shutil.rmtree(DEST_DIR)
os.makedirs(DEST_DIR)
for class_name in MASTER_CLASSES:
    os.makedirs(os.path.join(DEST_DIR, class_name), exist_ok=True)
print("✓ Destination cleaned and folders created.")

TOTAL_COPIED = 0
CLASS_COUNTS = {name: 0 for name in MASTER_CLASSES}

def copy_file(src, dest_class, file_id):
    """Helper function to copy files and avoid name collisions."""
    global TOTAL_COPIED
    try:
        # Get extension dynamically
        ext = os.path.splitext(src)[1]
        if not ext: ext = '.jpg' # Default if no extension

        filename = f"{dest_class}_{file_id}{ext}"
        dest_path = os.path.join(DEST_DIR, dest_class, filename)

        # Avoid re-copying if script is run multiple times
        if os.path.exists(dest_path):
            return

        shutil.copyfile(src, dest_path)
        TOTAL_COPIED += 1
        CLASS_COUNTS[dest_class] += 1
    except Exception as e:
        print(f"⚠️ Warning: Failed to copy {src}. Error: {e}")

# --- 3. PROCESS THE 'infrared_solar_modules' DATASET ---
print("\n" + "=" * 80)
print("Processing 1: 'infrared_solar_modules' (JSON)...")
print("=" * 80)
try:
    json_path = os.path.join(SOURCES_DIR, 'infrared_solar_modules', '2020-02-14_InfraredSolarModules', 'InfraredSolarModules', 'module_metadata.json')
    img_dir = os.path.join(SOURCES_DIR, 'infrared_solar_modules', '2020-02-14_InfraredSolarModules', 'InfraredSolarModules', 'images')

    if not os.path.exists(json_path):
         raise FileNotFoundError(f"JSON file not found at {json_path}")
    if not os.path.exists(img_dir):
         raise FileNotFoundError(f"Image directory not found at {img_dir}")

    with open(json_path, 'r') as f:
        metadata = json.load(f)

    print(f"✓ Loaded {len(metadata)} records. Pre-sorting files...")

    # --- Sort files before copying ---
    files_to_copy = {class_name: [] for class_name in MASTER_CLASSES}
    all_no_anomaly = []
    skipped_count = 0

    for image_key, data in metadata.items():
        original_class = data.get('anomaly_class')
        if original_class not in CLASS_MAP:
            skipped_count += 1
            continue

        target_class = CLASS_MAP[original_class]
        img_filename = os.path.basename(data.get('image_filepath', ''))
        if not img_filename:
            skipped_count += 1
            continue

        src_path = os.path.join(img_dir, img_filename)

        if not os.path.exists(src_path):
            # Try adding .jpg if missing extension in metadata
            src_path_jpg = src_path + '.jpg'
            if os.path.exists(src_path_jpg):
                src_path = src_path_jpg
            else:
                # print(f"  - Skipping {image_key}: Image file not found at {src_path} or {src_path_jpg}")
                skipped_count += 1
                continue

        file_id = f"infrared_{image_key}"

        if target_class == "no_anomaly":
            all_no_anomaly.append((src_path, target_class, file_id))
        else:
            files_to_copy[target_class].append((src_path, target_class, file_id))

    print(f"✓ Sorted {sum(len(v) for v in files_to_copy.values()) + len(all_no_anomaly)} potential images.")
    print(f"  (Skipped {skipped_count} invalid or unmapped entries)")

    # --- Undersample 'no_anomaly' ---
    # Aim for roughly 2x the average size of other classes, but cap at 2000
    avg_other_class_size = sum(len(files_to_copy[c]) for c in MASTER_CLASSES if c != 'no_anomaly') / (len(MASTER_CLASSES) - 1)
    target_no_anomaly_size = min(2000, max(500, int(avg_other_class_size * 2))) # Ensure at least 500

    print(f"\nBalancing 'no_anomaly':")
    print(f"  Average other class size: {avg_other_class_size:.0f}")
    print(f"  Found {len(all_no_anomaly)} 'no_anomaly' files.")
    print(f"  Target 'no_anomaly' size: {target_no_anomaly_size}")

    random.seed(42) # for reproducibility
    random.shuffle(all_no_anomaly)
    undersampled_no_anomaly = all_no_anomaly[:target_no_anomaly_size]
    files_to_copy["no_anomaly"] = undersampled_no_anomaly
    print(f"✓ Undersampled 'no_anomaly' to {len(undersampled_no_anomaly)} images.")

    # --- Copy all sorted files ---
    print("\nCopying files to master_dataset...")
    for class_name in MASTER_CLASSES:
        file_list = files_to_copy[class_name]
        print(f"  {class_name:15s}: Copying {len(file_list):5d} images...", end='', flush=True)
        for src, dest_class, file_id in file_list:
            copy_file(src, dest_class, file_id)
        print(" ✓ Done")

    print("\n✓ Dataset 1 processing complete.")

except Exception as e:
    print(f"\n❌ FAILED to process 'infrared_solar_modules'. Error: {e}")
    import traceback
    traceback.print_exc()


# --- 4. PROCESS DATASET 2: 'thermal_anomaly' (The YOLO one) ---
print("\n" + "=" * 80)
print("Processing 2: 'thermal_anomaly' (YOLO .txt)...")
print("=" * 80)
try:
    THERMAL_MAP = {
        0: "hotspot", 1: "hotspot", 2: "pid",
        3: "pid", 4: "pid", 5: "pid",
    }

    thermal_dir = os.path.join(SOURCES_DIR, 'thermal_anomaly')
    if not os.path.exists(thermal_dir):
         print("⚠️  'thermal_anomaly' dataset not found, skipping...")
    else:
        for split in ['train', 'valid', 'test']:
            img_folder = os.path.join(thermal_dir, 'ImageSet', split, 'images')
            lbl_folder = os.path.join(thermal_dir, 'ImageSet', split, 'labels')

            if not os.path.exists(img_folder) or not os.path.exists(lbl_folder):
                print(f"  Skipping {split} split (folders not found)")
                continue

            print(f"\n  Processing {split} split...")
            count = 0
            processed_labels = 0
            label_files = [f for f in os.listdir(lbl_folder) if f.endswith('.txt')]

            for lbl_file in label_files:
                processed_labels += 1
                # Try finding corresponding image (.jpg or .png)
                base_name = lbl_file.replace('.txt', '')
                img_path = None
                for ext in ['.jpg', '.png']:
                    potential_path = os.path.join(img_folder, base_name + ext)
                    if os.path.exists(potential_path):
                        img_path = potential_path
                        break

                if not img_path:
                    # print(f"    - Skipping label {lbl_file}: No matching image found.")
                    continue

                # Read labels
                try:
                    with open(os.path.join(lbl_folder, lbl_file), 'r') as f: lines = f.readlines()
                except Exception as read_e:
                    print(f"⚠️ Warning: Could not read label file {lbl_file}. Error: {read_e}")
                    continue

                found_classes_in_file = set()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) > 0:
                        try:
                            class_id = int(parts[0])
                            if class_id in THERMAL_MAP:
                                found_classes_in_file.add(THERMAL_MAP[class_id])
                        except ValueError:
                            continue # Skip malformed lines

                # Copy if we found relevant classes
                if len(found_classes_in_file) > 0:
                    # Prioritize rarer classes if multiple exist in one image
                    priority_order = ["pid", "hotspot"]
                    target_class = list(found_classes_in_file)[0] # Default to first found
                    for p_class in priority_order:
                       if p_class in found_classes_in_file:
                           target_class = p_class
                           break

                    file_id = f"thermal_{split}_{base_name}"
                    copy_file(img_path, target_class, file_id)
                    count += 1
            print(f"    ✓ Processed {processed_labels} labels, Copied {count} images")

        print("\n✓ Dataset 2 processing complete.")
except Exception as e:
    print(f"\n❌ FAILED to process 'thermal_anomaly'. Error: {e}")
    import traceback
    traceback.print_exc()

# --- 5. FINAL REPORT ---
print("\n" + "=" * 80)
print("MASTER DATASET BUILD COMPLETE!")
print("=" * 80)

print(f"\n✓ Successfully built dataset with {TOTAL_COPIED} images.")
print("\nFinal distribution in 'master_dataset':")

total_final = 0
max_count = 0
min_count = float('inf')

for class_name in MASTER_CLASSES:
    count = CLASS_COUNTS.get(class_name, 0)
    total_final += count
    if count > 0:
         max_count = max(max_count, count)
         min_count = min(min_count, count)
    status = "✅" if count > 0 else "❌"
    print(f"{status} {class_name:15s}: {count:5d} images")

print("-" * 35)
print(f"   {'TOTAL':15s}: {total_final:5d} images")

# Quality checks
print("\n" + "=" * 80)
print("QUALITY CHECKS")
print("=" * 80)

if total_final == 0:
     print("❌ CRITICAL ERROR: No images were copied. Check paths and permissions.")
elif total_final < 1000:
    print("⚠️  WARNING: Very small dataset (< 1000 images)")
    print("   Consider adding more data sources if possible.")
elif total_final < 3000:
    print("🟡 NOTICE: Dataset size is modest (1000-3000 images).")
    print("   Target accuracy: ~75-85%")
else:
    print("✅ Good dataset size (3000+ images).")
    print("   Target accuracy: 85%+ achievable")

if min_count == float('inf'): min_count = 0 # Handle case with empty classes

if max_count > min_count * 10 and min_count > 0:
    print("\n🔥 WARNING: Severe class imbalance!")
    print(f"   Ratio (Max/Min): {max_count / min_count:.1f}x (Max: {max_count}, Min: {min_count})")
    print("   → Training script *must* use Weighted Sampling.")
elif max_count > min_count * 5 and min_count > 0:
    print("\n🟡 NOTICE: Moderate class imbalance detected.")
    print(f"   Ratio (Max/Min): {max_count / min_count:.1f}x")
    print("   → Weighted sampling is recommended.")
elif total_final > 0:
    print("\n✅ Classes appear reasonably balanced.")

# Check for empty classes
empty_classes = [c for c in MASTER_CLASSES if CLASS_COUNTS.get(c, 0) == 0]
if empty_classes:
     print(f"\n❌ CRITICAL WARNING: The following classes have ZERO images: {', '.join(empty_classes)}")
     print("   The model will fail to train properly. Check CLASS_MAP and dataset sources.")


print("\n" + "=" * 80)
print("NEXT STEP: Run the training script")
print("=" * 80)
print("\n  python scripts/train.py\n")
