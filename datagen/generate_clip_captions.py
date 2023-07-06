import json
import os



def generate_clip_captions(dataset_folder):
    # load dataset info
    info = json.load(open(os.path.join(dataset_folder, "info.json"), "r"))

    tasks = list(info["tasks"].keys())
    print(self.tasks)
    print("num tasks:", len(self.tasks))
    split_index = int(len(self.tasks) * train_percentage)
    # self.tasks = train_tasks + val_tasks
    self.total_num_tasks = len(self.tasks)
    self.train_indices = np.arange(split_index)
    self.val_indices = split_index + np.arange(self.total_num_tasks - split_index)
    # self.train_percentage = train_percentage

    # task to task sample mapping, total_num_samples, and total_num_keyframes
    self.task_samples_start_indices = np.ndarray(shape=(self.total_num_tasks,), dtype=np.int64)
    self.task_num_samples = np.ndarray(shape=(self.total_num_tasks,), dtype=np.int64)
    task_samples_start_index = 0
    total_num_samples = 0
    total_num_keyframes = 0
    for i in range(self.total_num_tasks):
        task = self.tasks[i]
        self.task_samples_start_indices[i] = task_samples_start_index
        self.task_num_samples[i] = len(self.info["tasks"][task]["samples"])
        task_samples_start_index += self.task_num_samples[i]
        total_num_samples += self.task_num_samples[i]
        # we add +1 for the initial image
        total_num_keyframes += self.task_num_samples[i] * (len(self.info["tasks"][task]["instructions"]) + 1)

    self.image_features = torch.load(os.path.join(dataset_folder, "image_features.tar")).to(DEVICE)

    # initialize tensors for target data
    # TODO convert indices to one-hot
    # NOTE brick_{1,2} are not defined for the last keyframe
    self.brick_1 = torch.from_numpy(np.ndarray(shape=(total_num_keyframes,), dtype=np.int64)).to(DEVICE)
    self.brick_2 = torch.from_numpy(np.ndarray(shape=(total_num_keyframes,), dtype=np.int64)).to(DEVICE)
    # problem is not wee defined if we use pin_1_x and pin_2_x since, for example, pin_1_x = 2 and pin_2_x = 2 is
    # the same as pin_1_x = 1 and pin_2_x = 1 and so on
    # self.pin_1_x = torch.from_numpy(np.ndarray(shape=(total_num_keyframes,), dtype=np.int64)).to(DEVICE)
    # self.pin_2_x = torch.from_numpy(np.ndarray(shape=(total_num_keyframes,), dtype=np.int64)).to(DEVICE)
    self.pin_x = torch.from_numpy(np.ndarray(shape=(total_num_keyframes,), dtype=np.int64)).to(DEVICE)
    # self.pin_1_y = torch.from_numpy(np.ndarray(shape=(total_num_keyframes,), dtype=np.int64)).to(DEVICE)
    # self.pin_2_y = torch.from_numpy(np.ndarray(shape=(total_num_keyframes,), dtype=np.int64)).to(DEVICE)
    self.pin_y = torch.from_numpy(np.ndarray(shape=(total_num_keyframes,), dtype=np.int64)).to(DEVICE)
    # self.orientation = torch.from_numpy(np.ndarray(shape=(total_num_keyframes,), dtype=np.int64)).to(DEVICE)

    # colors
    color_to_object_id = {}
    for i, color in enumerate(self.info["dataset"]["color_keys"]):
        color_to_object_id[color] = i

    # task sample to keyframe mapping
    self.sample_idx_to_keyframes_start_index = np.ndarray(shape=(total_num_samples,), dtype=np.int64)
    self.sample_idx_to_num_instructions = np.ndarray(shape=(total_num_samples,), dtype=np.int64)

    # load data
    max_pin = self.info["dataset"]["max_pin"]
    sample_idx = 0
    for i in range(len(self.tasks)):
        task = self.tasks[i]
        task_i_samples = self.info["tasks"][task]["samples"]
        task_i_instructions = self.info["tasks"][task]["instructions"]
        for s in range(len(task_i_samples)):
            sample = task_i_samples[s]
            sample_keyframes_start_index = sample["keyframes_start_index"]
            sample_brick_colors = sample["brick_colors"]
            self.sample_idx_to_keyframes_start_index[sample_idx] = sample_keyframes_start_index
            self.sample_idx_to_num_instructions[sample_idx] = len(task_i_instructions)
            sample_idx += 1
            for ii in range(len(task_i_instructions)):
                brick_1_color = sample_brick_colors[task_i_instructions[ii]["brick_1"]]
                brick_2_color = sample_brick_colors[task_i_instructions[ii]["brick_2"]]
                self.brick_1[sample_keyframes_start_index + ii] = color_to_object_id[brick_1_color]
                self.brick_2[sample_keyframes_start_index + ii] = color_to_object_id[brick_2_color]
                # self.pin_1_x[sample_keyframes_start_index + ii] = task_i_instructions[ii]["pin_1_x"]
                # self.pin_2_x[sample_keyframes_start_index + ii] = task_i_instructions[ii]["pin_2_x"]
                # self.pin_1_y[sample_keyframes_start_index + ii] = task_i_instructions[ii]["pin_1_y"]
                # self.pin_2_y[sample_keyframes_start_index + ii] = task_i_instructions[ii]["pin_2_y"]
                self.pin_x[sample_keyframes_start_index + ii] = \
                    task_i_instructions[ii]["pin_1_x"] - task_i_instructions[ii]["pin_2_x"] + max_pin - 1
                self.pin_y[sample_keyframes_start_index + ii] = \
                    task_i_instructions[ii]["pin_1_y"] - task_i_instructions[ii]["pin_2_y"] + max_pin - 1
                # self.orientation[i] = task_i_instructions[ii]["orientation"]
    self.sample_idx_to_num_instructions = np.array(self.sample_idx_to_num_instructions)
    self.sample_idx_to_num_instructions = np.array(self.sample_idx_to_num_instructions)

    pass