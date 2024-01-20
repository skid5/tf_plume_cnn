#image loader
#resizes images to 64x64 and normalizess pixel values
def image_loader(image_path, y_size, x_size):
    with rasterio.open(image_path) as src:
        # Read NO2 data stored in the first band
        image_data = src.read(1)

    # Calculate the zoom factors for each dimension
    zoom_factor_y = y_size / image_data.shape[0]
    zoom_factor_x = x_size / image_data.shape[1]

    # Resize the array
    image = zoom(image_data, (zoom_factor_y, zoom_factor_x))
    #normalize the values
    if image.max() > 0:
        image = image/image.max()
    return image
    
#apply a random combination of shift and rotation
def invariant_transformation(images):
    shape = np.shape(images)
    r = np.floor(np.random.rand(shape[0])*360)
    s = np.random.randint(-3,3,size=shape[0])
    transformed_images = np.zeros(shape)
    for i in range(shape[0]):
        transformed_images[i] = scipy.ndimage.shift(scipy.ndimage.rotate(images[i],r[i],reshape=False), s[i])
        if transformed_images[i].max() > 0:
            transformed_images[i] = transformed_images[i]/transformed_images[i].max()
    return transformed_images