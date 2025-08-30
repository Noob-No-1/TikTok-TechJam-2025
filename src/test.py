from image_augmented_classifier import classify_review_with_image

res = classify_review_with_image(
    category="Pizza restaurant",
    rating=2,
    text="The dough was soggy but service was friendly.",
    image_path="image/Lays_advertisement-examples.jpg",
    place_name="Mario's Slice",
)
print(res)  # should include _image_caption and _image_heuristics