# class, color, bbox_3d, occ

class_one = ["Identify all objects belonging to class A*.", 
             "Find every instance classified as class A*.", 
             "Retrieve all items labeled as class A*.", 
             "Every object belonging to class A* should be found.",
             "Detect every entity classified as class A*.", 
             "Collect all instances of class A*.", 
             "Gather all objects that are classified as class A*.", 
             "Select every item that are class A*.", 
             "All objects in class A* must be located.", 
             "All class A* need to be identified.",] 
class_one_substitute_word = ["class A*"]


class_one_color_one = ["Detect B*-colored objects from the A*.", 
                       "Search for B*-colored objects within the class A*.", 
                       "Locate objects that are of color B* and belong to class A*.",
                       "Look for B*-colored items within class A*.",
                       "Retrieve instances of class A* that are B*-colored.", 
                       "Find objects that are both class A* and B*-colored.",
                       "Pick out B*-colored A*.",
                       "Seek objects in class A* which have a color B*.",]
class_one_color_one_substitute_word = ["class A*", "A*", "B*"]


class_color_bbox_distance = ["Identify and segment the object in class B* that is howfar from the object in class A* that has color C*.",
                            "Find the object in class B* that is the howfar to the object with color C* in class A*.",
                            "Extract the object from class B* that is positioned howfar to the colored C* object within class A*.",
                            "Segment the object in class B* that lies at the howfar distance from the C*-colored object in class A*.", 
                            "Pick out the object from class B* that is the howfar distant from the C*-colored object in class A*, and isolate it.",
                            "Locate and segment the object in class B* that is howfar to the C*-colored object in class A*.",
                            "Segment out the object in class B* that is positioned at the howfar distance from the object of color C* in class A*.",
                            "Identify the object in class B* that is howfar from the object in class A* of color C* and segment it.",
                            "Select and extract the object in class B* with the howfar distance to the C* object in the class A*.",
                            "Isolate the object in class B* that is howfar from the object in class A* that has the color C*.",]
class_color_bbox_diatance_howfar_1 = ["closest", "nearest"]  # 最近
class_color_bbox_diatance_howfar_2 = ["farthest", "most distant", "most remote"]  # 最远
class_color_bbox_diatance_howfar_3 = ["second closest", "next closest", "second nearest"]  # 第二近
class_color_bbox_diatance_howfar_4 = ["second farthest", "next farthest", "second remotest"]  # 第二远
class_color_distance_substitute_word = ["class A*", "class B*", "C*", "howfar"]


class_color_bbox_size = ["Identify the howsize object from the set of class A* items that are colored C*.", 
                        "Please select the howsize object among all class A* items with the color C*.", 
                        "Extract the howsize item within the group of objects that belong to class A* and have color C*.",
                        "Highlight the howsize object from the collection of C*-colored objects in class A*.", 
                        "Separate the howsize C* object belonging to class A*.", 
                        "Mark the howsize object in the group of class A* that are colored C*.", 
                        "Find and segment the howsize object among the objects that are both class A* and have color C*.",
                        "Locate the howsize C* object among the items of class A*.",
                        "Select and segment the howsize C* object belonging to class A*.",
                        "Choose the howsize item among the objects that are of class A* and are colored C*.",]
class_color_bbox_size_howsize_1 = ["largest", "most sizable", "biggest"]  # 最大
class_color_bbox_size_howsize_2 = ["smallest", "tiniest"]  # 最小
class_color_bbox_size_howsize_3 = ["second largest", "next to largest", "second biggest"]  # 第二大
class_color_bbox_size_howsize_4 = ["second smallest", "next to smallest", "second tiniest"]  # 第二小
class_color_bbox_size_substitute_word = ["class A*", "C*", "howsize"]


implicit_class_one = {
    "cabinet": [
        "Furniture used to store various items, but does not have a freezer function.",
        "A structure with compartments or shelves used to keep items organized, such as clothes.",
        "Typically placed in rooms to hold belongings, such as hats and quilts, securely and neatly."
    ],
    "bed": [
        "Furniture used for sleeping at night.",
        "A piece of furniture with a mattress for rest.",
        "The main place where you sleep."
    ],
    "chair": [
        "A seat for one person with a backrest.",
        "Furniture you sit on individually.",
        "A single-person seat, not for lying down."
    ],
    "sofa": [
        "A long, comfortable seat for multiple people.",
        "Furniture in the living room for several people to sit.",
        "A large, cushioned seat for family seating."
    ],
    "table": [
        "A flat surface on legs generally used for play, or dining and typically has no drawers.",
        "Furniture where you often gather for eating or play, typically without drawers.",
        "A surface supported by legs for dining or play, usually lacking drawers."
    ],
    "door": [
        "A movable barrier designed to open and close an entrance, such as to a room.",
        "An object used to enter or exit a room.",
        "A structure that provides an entrance to a room or building."
    ],
    "window": [
        "An opening in a wall with glass to see outside.",
        "A transparent panel with glass to let in light and air.",
        "A wall opening with glass that allows daylight inside."
    ],
    "bookshelf": [
        "Furniture with shelves to store books.",
        "A set of shelves for organizing books.",
        "A place to keep and display books."
    ],
    "picture": [
        "An image or artwork hung on a wall.",
        "A framed photo or painting for decoration.",
        "An artistic representation displayed on walls."
    ],
    "counter": [
        "A flat surface often in kitchens, bathrooms, and bars, used for food preparation or holding items.",
        "Typically found in kitchens, bathrooms, and bars, a surface is ideal for food preparation and quick storage.",
        "A long countertop commonly in kitchens, bars, and bathrooms, perfect for meal preparation and temporary use."
    ],
    "desk": [
        "Furniture with a flat top and drawers for work or study.",
        "A piece of furniture used for writing or computer work.",
        "A workstation with storage, used in offices."
    ],
    "curtain": [
        "Fabric hung to block light or provide privacy.",
        "Cloth used to block light from a room.",
        "Cloth used to reduce sunlight or control brightness in a room."
    ],
    "refrigerator": [
        "An appliance that keeps food and drinks cold.",
        "A device used to cool and preserve food.",
        "A furniture for storing perishable items."
    ],
    "shower curtain": [
        "A waterproof barrier hung to prevent water from splashing out during bathing.",
        "A protective pendant used to block water in a bathing area.",
        "A hanging screen that keeps water inside the bathing space."
    ],
    "toilet": [
        "A fixture used for disposing of human waste.",
        "A bathroom unit you sit on for excretion.",
        "A bowl with a seat for bodily waste disposal."
    ],
    "sink": [
        "A basin with a tap for washing hands or dishes.",
        "A fixture with a faucet and drain for washing.",
        "A bowl-shaped unit used to hold water for cleaning."
    ],
    "bathtub": [
        "A large container for bathing.",
        "A tub used for soaking and washing the body.",
        "A fixture in the bathroom for immersion bathing."
    ]
}


implicit_more_classes = [
    {"Question": "Furniture used for sitting or lying down.", "Answer": ["bed", "chair", "sofa"]},
    {"Question": "Items used for bathing.", "Answer": ["shower curtain", "sink", "bathtub"]},
    {"Question": "Appliances or furniture used to store food.", "Answer": ["cabinet", "refrigerator"]},
    {"Question": "Fixtures used for personal hygiene.", "Answer": ["toilet", "sink", "bathtub"]},
    {"Question": "Furniture used to display books or clothes.", "Answer": ["cabinet", "bookshelf"]},
    {"Question": "Furniture used for resting.", "Answer": ["bed", "chair", "sofa"]},
    {"Question": "Items used to decorate walls.", "Answer": ["bookshelf", "picture"]},
    {"Question": "Furniture that people use to organize their clothes and personal belongings.", "Answer": ["cabinet", "bookshelf", "desk"]},
    {"Question": "Objects in the bathroom that provide privacy and separate wet and dry areas.", "Answer": ["shower curtain", "bathtub"]},
    {"Question": "Items used in the bathroom to aid bathing and maintain hygiene.", "Answer": ["sink", "shower curtain", "toilet", "bathtub"]},
]

