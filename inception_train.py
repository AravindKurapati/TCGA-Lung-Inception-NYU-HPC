import argparse
import numpy as np
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tensorflow.keras import layers, models, optimizers, losses, callbacks
from tensorflow.keras.applications import InceptionV3
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from barlow_twins_module import BarlowTwins, ResNetBackbone
import torchvision.transforms.functional as TF
import matplotlib
matplotlib.use('Agg')
import os
print(tf.__version__)

# Argument Parser
parser = argparse.ArgumentParser(description='Train Inception model on TCGA lung data.')
parser.add_argument('--epochs', type=int, default=90, help='Number of epochs to run, default is 90 epochs.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size, reduced to fit memory.')
parser.add_argument('--img_size', type=int, default=224, help='Image size for the model.')
parser.add_argument('--checkpoint_dir', type=str, required=True, help='Directory to save checkpoints.')
parser.add_argument('--model_dir', type=str, required=True, help='Directory to save the final trained model.')
parser.add_argument('--test_hdf5', type=str, required=True, help='Path to test HDF5 file')
parser.add_argument('--labels_txt', type=str, required=True, help='Path to the labels file')
parser.add_argument('--hpc_ids_csv', type=str, required=True, help='Path to the CSV file with HPC IDs') 
args = parser.parse_args()
tf.keras.backend.set_image_data_format('channels_last')

#loading data from HDF5
def load_hdf5_data(hdf5_path, img_key, slide_key, tile_key):
    with h5py.File(hdf5_path, 'r') as h5_file:
        images = np.array(h5_file[img_key])
        slides = np.array(h5_file[slide_key])
        tiles = np.array(h5_file[tile_key])
    return images, slides, tiles



#loading test data
test_images, test_slides, test_tiles = load_hdf5_data(args.test_hdf5, 'img_z_latent', 'slides', 'tiles')
test_images = test_images / 255.0  



def get_patient_id(slide_id):    
    patient_id = slide_id[:12]
#     print(f"Extracted Patient ID: {patient_id} from Slide ID: {slide_id}")
    return patient_id



#patient labels
def load_patient_labels(label_file):
    patient_to_label = {}
    with open(label_file, 'r') as file:
        for line in file:
            patient_id, label = line.strip().split()
            patient_to_label[patient_id[:12]] = 1 if 'LUAD' in label else 0
    print("Loaded patient labels. Sample data:", list(patient_to_label.items())[:15])
    return patient_to_label

#getting labels for slides
patient_to_label = load_patient_labels(args.labels_txt)


#labels for slides with debug
def get_labels_for_slides(slides, patient_to_label):
    labels = []
    unmapped_ids = []  

    for i, slide in enumerate(slides):
        slide_id = slide.decode("utf-8")[:12]  
        label = patient_to_label.get(slide_id, None)  

        if label is None:
            unmapped_ids.append(slide_id)
        
        labels.append(label)

        if i < 20:
            print(f"Slide ID: {slide_id}, Assigned Label: {label}")

    print(f"Completed label assignment for {len(labels)} slides.")
    print(f"Unmapped Slide IDs: {len(unmapped_ids)} - {unmapped_ids[:10]} (First 10)")
    return np.array(labels, dtype=object)




#mapping of patient IDs to labels
def map_patient_labels(slides, labels):
    patient_to_label_map = {}
    print("Starting to map patient IDs to labels...")
    
    for slide, label in zip(slides, labels):
        slide_id = slide.decode("utf-8") if isinstance(slide, bytes) else slide
        patient_id = slide_id[:12]  
        
        if patient_id not in patient_to_label_map:
            patient_to_label_map[patient_id] = label
        elif patient_to_label_map[patient_id] != label:
            print(f"Warning: Inconsistent labels for Patient ID {patient_id}. Previous: {patient_to_label_map[patient_id]}, New: {label}")
        
    print(f"Completed mapping of {len(patient_to_label_map)} patient IDs to labels.")
    return patient_to_label_map


#generating slide labels directly from test_slides
slide_labels = get_labels_for_slides(test_slides, patient_to_label)

#generating the patient_to_label_map using all test slides and their labels
patient_to_label_map = map_patient_labels(test_slides, slide_labels)

#generating patient IDs and their labels
unique_patient_ids = np.array(list(patient_to_label_map.keys()))
patient_labels = np.array(list(patient_to_label_map.values()))

#split dataset by patient
train_patient_ids, temp_patient_ids, train_patient_labels, temp_patient_labels = train_test_split(
    unique_patient_ids, patient_labels, test_size=0.4, random_state=42, stratify=patient_labels)

valid_patient_ids, test_patient_ids, valid_patient_labels, test_patient_labels = train_test_split(
    temp_patient_ids, temp_patient_labels, test_size=0.5, random_state=42, stratify=temp_patient_labels)

#applying train, validation, and test masks directly on test_slides
train_mask = np.isin([get_patient_id(slide.decode("utf-8")) for slide in test_slides], train_patient_ids)
valid_mask = np.isin([get_patient_id(slide.decode("utf-8")) for slide in test_slides], valid_patient_ids)
test_mask = np.isin([get_patient_id(slide.decode("utf-8")) for slide in test_slides], test_patient_ids)

#filtering data using masks
train_images, train_slides, train_tiles = test_images[train_mask], test_slides[train_mask], test_tiles[train_mask]
valid_images, valid_slides, valid_tiles = test_images[valid_mask], test_slides[valid_mask], test_tiles[valid_mask]
test_images, test_slides, test_tiles = test_images[test_mask], test_slides[test_mask], test_tiles[test_mask]

#unique slides in each dataset
train_unique_slides = np.unique(train_slides)
valid_unique_slides = np.unique(valid_slides)
test_unique_slides = np.unique(test_slides)

#labels for unique slides
train_labels = np.array([patient_to_label_map[get_patient_id(slide.decode("utf-8"))] for slide in train_unique_slides])
valid_labels = np.array([patient_to_label_map[get_patient_id(slide.decode("utf-8"))] for slide in valid_unique_slides])
test_labels = np.array([patient_to_label_map[get_patient_id(slide.decode("utf-8"))] for slide in test_unique_slides])


#data from TFRecords
def create_tfrecord_dataset(tfrecord_path, batch_size, is_training=True):
    def parse_example(example_proto):
        features = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }
        parsed_features = tf.io.parse_single_example(example_proto, features)
        image = tf.io.parse_tensor(parsed_features['image'], out_type=tf.float32)
        label = tf.cast(parsed_features['label'], tf.int32)
        image = tf.reshape(image, [args.img_size, args.img_size, 128])
        return image, label

    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_example)  
    if is_training:
        dataset = dataset.shuffle(1000).batch(batch_size).repeat()
    else:
        dataset = dataset.batch(batch_size)
    return dataset



#extract x and y coordinates from tile string
def extract_coordinates(tile_str):
    tile_str = tile_str.decode("utf-8")  
    tile_str = tile_str.replace('.jpeg', '').replace('.jpg', '')  
    x, y = map(int, tile_str.split('_'))  
    return x, y

def reconstruct_images(images, slides, tiles, img_size, channels):
    unique_slides = np.unique(slides)
    slide_to_idx = {slide.decode("utf-8"): idx for idx, slide in enumerate(unique_slides)}
    reconstructed = np.zeros((len(unique_slides), img_size, img_size, channels), dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        x, y = extract_coordinates(tile)
        slide_id = slides[idx].decode("utf-8")
        slide_idx = slide_to_idx[slide_id]
        reconstructed[slide_idx, x, y] = images[idx]
    return reconstructed


img_size = args.img_size
channels = 128  

train_reconstructed = reconstruct_images(train_images, train_slides, train_tiles, img_size, channels)
valid_reconstructed = reconstruct_images(valid_images, valid_slides, valid_tiles, img_size, channels)
test_reconstructed = reconstruct_images(test_images, test_slides, test_tiles, img_size, channels)


data_augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal_and_vertical'),
    layers.RandomRotation(0.2),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.2),
])

    

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))
    
def serialize_example(image, label):
    image = tf.cast(image, tf.float32)  
    feature = {
        'image': _bytes_feature(image),
        'label': _int64_feature(label)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def write_tfrecords(filename, images, labels):
    with tf.io.TFRecordWriter(filename) as writer:
        for image, label in zip(images, labels):
            augmented_img = data_augmentation(tf.expand_dims(image, axis=0))
            tf_example = serialize_example(augmented_img[0], label)
            writer.write(tf_example)
    print(f"TFRecord file saved at: {filename}")

def extract_patch_embeddings(images, embedding_dim=128):
    return np.random.rand(images.shape[0], embedding_dim).astype(np.float32)

train_patch_embeddings = extract_patch_embeddings(train_reconstructed)
valid_patch_embeddings = extract_patch_embeddings(valid_reconstructed)
test_patch_embeddings = extract_patch_embeddings(test_reconstructed)

class TileDataset(Dataset):
    def __init__(self, tiles, transform=None):
        self.tiles = tiles
        self.transform = transform

    def __getitem__(self, idx):
        tile = self.tiles[idx]
        x1 = self.transform(tile)
        x2 = self.transform(tile)
        return x1, x2

    def __len__(self):
        return len(self.tiles)
barlow_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
])


def train_barlow_twins(tiles_np, batch_size=64, epochs=10, device='cuda'):
    # Convert NumPy array to torch dataset
    dataset = TileDataset(tiles_np, transform=barlow_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    backbone = ResNetBackbone().to(device)
    model = BarlowTwins(backbone).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x1, x2 in dataloader:
            x1, x2 = x1.to(device), x2.to(device)
            loss = model(x1, x2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss / len(dataloader):.4f}")

    return backbone

def extract_representations(backbone, tiles_np, batch_size=64, device='cuda'):
    backbone.eval()
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(tiles_np), batch_size):
            batch = [transform(img) for img in tiles_np[i:i+batch_size]]
            batch = torch.stack(batch).to(device)
            emb = backbone(batch).cpu().numpy()
            embeddings.append(emb)

    return np.vstack(embeddings)
bt_backbone = train_barlow_twins(train_reconstructed, epochs=10, device=device)

train_embeddings = extract_representations(bt_backbone, train_reconstructed, device=device)

train_tfrecord_path = os.path.join(args.model_dir, 'train_data.tfrecord')
valid_tfrecord_path = os.path.join(args.model_dir, 'valid_data.tfrecord')

write_tfrecords(train_tfrecord_path, train_reconstructed, train_labels[:len(train_reconstructed)])
write_tfrecords(valid_tfrecord_path, valid_reconstructed, valid_labels[:len(valid_reconstructed)])

    
train_dataset = create_tfrecord_dataset(train_tfrecord_path, args.batch_size, is_training=True)
valid_dataset = create_tfrecord_dataset(valid_tfrecord_path, args.batch_size, is_training=False)



    
def create_original_inception_model(input_shape=(224, 224, 128)):
    base_model = InceptionV3(include_top=False, weights=None, input_shape=input_shape)
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)  
    
    return models.Model(inputs=base_model.input, outputs=outputs)

model = create_original_inception_model((args.img_size, args.img_size, 128))
model.compile(optimizer=optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
              loss=losses.BinaryCrossentropy(), metrics=['accuracy'])
model.summary()











# for slide in test_slides[:10]:  
#     slide_id = slide.decode("utf-8")[:12]
#     print(f"Slide ID: {slide.decode('utf-8')}, Extracted Patient ID: {slide_id}")


# with open(args.labels_txt, 'r') as file:
#     labels_data = [line.strip().split()[0][:12] for line in file]
# print("Sample Patient IDs from labels_txt:", labels_data[:10])


# patient_id_cleaned = slide_id.strip()

# # Extract patient IDs from test slides
# test_patient_ids = {get_patient_id(slide.decode("utf-8")) for slide in test_slides}

# # Extract patient IDs from labels_txt
# label_patient_ids = set(patient_to_label.keys())

# missing_patient_ids = test_patient_ids - label_patient_ids

# print(f"Patient IDs in test slides but missing in labels_txt: {missing_patient_ids}")
# print(f"Number of missing IDs: {len(missing_patient_ids)}")


# print(f"Train labels distribution: {np.unique(train_patient_labels, return_counts=True)}")
# print(f"Validation labels distribution: {np.unique(valid_patient_labels, return_counts=True)}")
# print(f"Test labels distribution: {np.unique(test_patient_labels, return_counts=True)}")



# unique_patient_ids = [get_patient_id(slide.decode("utf-8")) for slide in test_slides]
# unique_labels = [patient_to_label_map.get(pid, 'Unknown') for pid in unique_patient_ids]
# print(f"Label distribution in test HDF5: {pd.Series(unique_labels).value_counts()}")


# train_patient_ids, _, train_patient_labels, _ = train_test_split(unique_patient_ids, unique_labels, test_size=0.4, random_state=42, stratify=unique_labels)
# print("Train set label distribution:", pd.Series(train_patient_labels).value_counts())

# h5_patient_ids = set([get_patient_id(slide.decode("utf-8")) for slide in test_slides])
# txt_patient_ids = set(patient_to_label.keys())
# missing_in_labels_txt = h5_patient_ids - txt_patient_ids
# print("Patient IDs in HDF5 but missing from labels_txt:", missing_in_labels_txt)


# def load_mask(mask_dir, slide_id):
#     mask_path = f"{mask_dir}/{slide_id}_mask.png"
#     if not os.path.exists(mask_path):
#         raise FileNotFoundError(f"Mask not found for slide ID {slide_id} at {mask_path}")
#     return plt.imread(mask_path)

# mask_dir = "mask_images"

# available_masks = [f for f in os.listdir(mask_dir) if f.endswith('_mask.png')]
# print("Sample test slides:", [slide.decode("utf-8") for slide in test_slides[:10]])
# print("Sample available masks:", available_masks[:10])


#validate if all test slides have corresponding masks
# unmatched_slides = [slide.decode("utf-8") for slide in test_slides if f"{slide.decode('utf-8')}_mask.png" not in available_masks]
# print("Unmatched Slide IDs:", unmatched_slides[:10])  # Display a sample
# print(f"Total unmatched slides: {len(unmatched_slides)}")


# from sklearn.cluster import DBSCAN

# def extract_coordinates_from_mask(mask):
#     coordinates = np.column_stack(np.nonzero(mask))
#     return coordinates

# def cluster_coordinates(coordinates, eps=5, min_samples=10):
#     clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coordinates)
#     return clustering.labels_

# slide_id = "TCGA-33-4532-01Z-00-DX1"  
# mask = load_mask(mask_dir, slide_id)
# coordinates = extract_coordinates_from_mask(mask)
# cluster_labels = cluster_coordinates(coordinates)

# print(f"Clusters for slide {slide_id}: {np.unique(cluster_labels)}")


# def save_clusters_visualization(coordinates, labels, slide_id, output_dir="cluster_visualizations"):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     plt.figure(figsize=(10, 10))
#     scatter = plt.scatter(coordinates[:, 1], coordinates[:, 0], c=labels, cmap='tab20', s=5)
#     plt.title(f"HPC Clusters for Slide {slide_id}")
#     plt.gca().invert_yaxis()
#     plt.colorbar(scatter, label="Cluster ID")
    
#     output_path = os.path.join(output_dir, f"{slide_id}_clusters.png")
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     plt.close()  # Close the plot to free memory
#     print(f"Cluster visualization saved for {slide_id} at {output_path}")


# slide_id = "TCGA-33-4532-01Z-00-DX1"  
# mask = load_mask(mask_dir, slide_id)
# coordinates = extract_coordinates_from_mask(mask)
# cluster_labels = cluster_coordinates(coordinates)
# save_clusters_visualization(coordinates, cluster_labels, slide_id)


# for slide in test_slides:
#     slide_id = slide.decode("utf-8")[:12]
#     try:
#         mask = load_mask(mask_dir, slide_id)
#         coordinates = extract_coordinates_from_mask(mask)
#         cluster_labels = cluster_coordinates(coordinates)
#         save_clusters_visualization(coordinates, cluster_labels, slide_id)
#     except FileNotFoundError as e:
#         print(e)


train_loss_results = []
valid_loss_results = []

early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
steps_per_epoch = len(train_reconstructed) // args.batch_size
validation_steps = len(valid_reconstructed) // args.batch_size

history = model.fit(
    train_dataset, 
    epochs=args.epochs, 
    steps_per_epoch=steps_per_epoch,
    validation_data=valid_dataset, 
    validation_steps=validation_steps, 
    callbacks=[early_stopping]
)

    
    

for epoch in range(len(history.history['loss'])):
    train_loss_results.append(history.history['loss'][epoch])
    valid_loss_results.append(history.history['val_loss'][epoch])

losses_df = pd.DataFrame({
    'epoch': range(1, len(train_loss_results) + 1),
    'train_loss': train_loss_results,
    'valid_loss': valid_loss_results
})
    
    
losses_csv_path = os.path.join(args.model_dir, 'training_validation_losses.csv')
losses_df.to_csv(losses_csv_path, index=False)
print(f"Training and validation losses saved at {losses_csv_path}")


if len(np.unique(valid_labels)) > 1:
    valid_predictions = model.predict(valid_reconstructed).ravel()
    valid_auc = roc_auc_score(valid_labels, valid_predictions)
    print(f"Validation AUC: {valid_auc}")
else:
    print("Validation labels contain only one class")

if len(np.unique(test_labels)) > 1:
    test_predictions = model.predict(test_reconstructed).ravel()
    test_auc = roc_auc_score(test_labels, test_predictions)
    print(f"Test AUC: {test_auc}")
else:
    print("Test labels contain only one class")

model_save_path = os.path.join(args.model_dir, 'trained_model.h5')
model.save(model_save_path)
print(f"Trained model saved at: {model_save_path}")

