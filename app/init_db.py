from models import app, db

def create_db():
    with app.app_context():
       db.create_all()
       print("Database and tables created successfully.")

def remove_db():
    with app.app_context():
        db.drop_all()
        print("Database and tables removed successfully.")
        
if __name__ == "__main__":
    while True:
        print("1. Create Database")
        print("2. Remove Database")
    
        choice = input("Enter your choice: ")
    
        if choice == "1":
            create_db()
        
        elif choice == "2":
           remove_db()