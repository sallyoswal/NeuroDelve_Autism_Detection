import tensorflow as tf
import os

class CreateTFRecord:
    def __init__(self, labels):
        self.labels = labels

    def convert_image_folder(self, img_paths, tfrecord_file_name):
        with tf.io.TFRecordWriter(tfrecord_file_name) as writer:
            total_images = 0
            for idx in range(len(img_paths)):
            example = self._convert_image(idx, img_paths[idx])
            writer.write(example.SerializeToString())
            total_images += 1

            print(f"Image number: {idx}")
            

    def _convert_image(self, idx, img_path):
        label = self.labels[idx]
        # Convert image to string data
        image_data = np.load(img_path)
        image_data = ((image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))).astype('float32')
        image_str = image_data.tobytes()
        
        example = tf.train.Example(features = tf.train.Features(feature = {
            'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [image_str])),
            'label': tf.train.Feature(int64_list = tf.train.Int64List(value = [label]))
            }))
        return example

# if __name__ == '__main__':
#     labels = train_labels
#     t = CreateTFRecord(labels)
#     t.convert_image_folder(img_paths, "/content/drive/MyDrive/Prediction of Autism /Data/TFRecorder/Train_img_lab.tfrecord")