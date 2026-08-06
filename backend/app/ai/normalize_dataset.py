import os
import csv
dataset_path="../dataset"

for letter in sorted(os.listdir(dataset_path)):
    folder=os.path.join(dataset_path, letter)

    if not os.path.isdir(folder):
        continue
    csv_path=os.path.join(folder, f"{letter}.csv")
    print(f"Normalizing {csv_path}...")

    samples=[]
    with open(csv_path, "r") as file:
        reader=csv.reader(file)

        for row in reader:
            sample=[float(value) for value in row]
            wrist_x=sample[0]
            wrist_y=sample[1]
            wrist_z=sample[2]
            normalized_sample=[]
            for i in range(0, len(sample), 3):#starting at i move 3 at a time
                normalized_sample.extend([
                    sample[i]-wrist_x,
                    sample[i+1]-wrist_y,
                    sample[i+2]-wrist_z
                ])
            samples.append(normalized_sample)

    with open(csv_path, "w", newline="") as file:
        writer=csv.writer(file)
        writer.writerows(samples)

    print(f"Finished normalizing {letter}")