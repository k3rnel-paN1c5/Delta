def infer(model, image_url):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = model(input_tensor)
            
        prediction = prediction.squeeze().cpu().numpy()
        
        # Normalize for visualization
        output_normalized = (prediction - prediction.min()) / (prediction.max() - prediction.min())
        output_image = Image.fromarray((output_normalized * 255).astype(np.uint8))
        
        print(f"Inference successful for {image_url}")
        output_image.save("depth_prediction.png")
        print("Prediction saved to depth_prediction.png")
        
    except Exception as e:
        print(f"Could not process image {image_url}: {e}")
