class UnlabeledImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(('png', 'jpg', 'JPG'))]
        print(f"Found {len(self.image_paths)} images in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert('RGB')

            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Could not load image {self.image_urls[idx]}: {e}")
            # Return a dummy tensor if an image fails to load
            return torch.zeros(3, 256, 256)
