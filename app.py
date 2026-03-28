from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__, static_folder='static')

model = None
scaler = None
le_dict = None
model_features = ['Food_Item', 'Quantity', 'Category_Encoded', 'Obesity_Risk', 'Diabetes_Risk', 'Cholesterol_Risk']

food_items = [
    "Chicken Breast", "Coke", "Apple", "Burger", "Rice", "Salmon", "Broccoli", 
    "Banana", "Egg", "Pizza", "Pasta", "Steak", "Salad", "Bread", "Milk",
    "Orange", "Carrot", "Potato", "Cheese", "Yogurt", "Almonds", "Avocado",
    "Turkey", "Tuna", "Shrimp", "Lobster", "Crab", "Oysters", "Scallops",
    "Clams", "Mussels", "Cod", "Tilapia", "Sardines", "Anchovies", "Mackerel",
    "Bass", "Trout", "Catfish", "Pork", "Lamb", "Duck", "Goose", "Venison",
    "Bison", "Rabbit", "Liver", "Kidney", "Heart", "Tongue", "Tripe",
    "Sausage", "Bacon", "Ham", "Pepperoni", "Salami", "Mortadella", "Prosciutto",
    "Corn", "Peas", "Beans", "Lentils", "Chickpeas", "Soybeans", "Tofu",
    "Tempeh", "Seitan", "Quinoa", "Barley", "Oats", "Wheat", "Rye", "Buckwheat",
    "Millet", "Bulgur", "Couscous", "Noodles", "Bagel", "Croissant", "Muffin", 
    "Pancake", "Waffle", "Cereal", "Granola", "Popcorn", "Pretzel", "Crackers", 
    "Chips", "Salsa", "Guacamole", "Hummus", "Olive Oil", "Butter", "Margarine", 
    "Mayonnaise", "Ketchup", "Mustard", "Relish", "Pickles", "Olives", "Capers", 
    "Hot Sauce", "BBQ Sauce", "Soy Sauce", "Fish Sauce", "Oyster Sauce", 
    "Hoisin Sauce", "Peanut Butter", "Jam", "Honey", "Maple Syrup", "Sugar", 
    "Chocolate", "Ice Cream", "Cake", "Cookies", "Brownies", "Pie", "Donut",
    "Pastry", "Candy", "Gum", "Soda", "Juice", "Coffee", "Tea",
    "Beer", "Wine", "Whiskey", "Vodka", "Rum", "Gin", "Tequila", "Brandy"
]

categories = {
    0: "Fast Food",
    1: "Dairy",
    2: "Beverages",
    3: "Protein",
    4: "Fruits",
    5: "Vegetables",
    6: "Grains",
    7: "Snacks"
}

def load_model():
    global model, scaler, le_dict
    try:
        with open('calorie_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('encoders.pkl', 'rb') as f:
            le_dict = pickle.load(f)
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def predict_calories(food_item, quantity, protein, carbohydrates, fat, category_encoded, obesity_risk, diabetes_risk, cholesterol_risk):
    global model, scaler, le_dict
    
    if model is None or scaler is None or le_dict is None:
        return None
    
    try:
        input_data = {
            'Food_Item': [food_item],
            'Quantity': [quantity],
            'Category_Encoded': [category_encoded],
            'Obesity_Risk': [obesity_risk],
            'Diabetes_Risk': [diabetes_risk],
            'Cholesterol_Risk': [cholesterol_risk]
        }
        
        sample_df = pd.DataFrame(input_data)
        
        if 'Food_Item' in le_dict and le_dict['Food_Item'] is not None:
            try:
                if food_item in le_dict['Food_Item'].classes_:
                    sample_df['Food_Item'] = le_dict['Food_Item'].transform(sample_df['Food_Item'])
                else:
                    sample_df['Food_Item'] = 0
            except Exception as e:
                print(f"Encoding error: {e}")
                sample_df['Food_Item'] = 0
        
        sample_df_aligned = sample_df[model_features]
        sample_scaled = scaler.transform(sample_df_aligned)
        prediction = model.predict(sample_scaled)
        
        return float(prediction[0])
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return None

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/food-items')
def get_food_items():
    return jsonify({
        'food_items': food_items,
        'categories': categories
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500
    
    try:
        result = predict_calories(
            food_item=str(data.get('food_item', 'Apple')),
            quantity=int(data.get('quantity', 1)),
            protein=float(data.get('protein', 0)),
            carbohydrates=float(data.get('carbohydrates', 0)),
            fat=float(data.get('fat', 0)),
            category_encoded=int(data.get('category_encoded', 0)),
            obesity_risk=float(data.get('obesity_risk', 5)),
            diabetes_risk=float(data.get('diabetes_risk', 5)),
            cholesterol_risk=float(data.get('cholesterol_risk', 5))
        )
    except Exception as e:
        print(f"Route error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 400
    
    if result is not None:
        return jsonify({'success': True, 'calories': round(result, 2)})
    else:
        return jsonify({'success': False, 'error': 'Prediction failed'}), 400

if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
