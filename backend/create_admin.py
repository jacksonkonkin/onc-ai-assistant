"""
Admin Account Creation Script
Run this script to create a pre-made admin account in the database.
"""

import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection
MONGO_URI = os.getenv("MONGO_URI")
client = AsyncIOMotorClient(MONGO_URI)
db = client["oncai"]
users_collection = db["users"]

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

async def create_admin_account():
    """Create a default admin account"""
    
    # Admin account details
    admin_data = {
        "username": "admin",
        "email": "admin@oncai.com",
        "password": "admin123",  # You should change this!
        "role": "admin",
        "is_indigenous": False,
        "onc_token": "admin-dev-token"
    }
    
    # Check if admin already exists
    existing_admin = await users_collection.find_one({"username": admin_data["username"]})
    if existing_admin:
        print(f"❌ Admin user '{admin_data['username']}' already exists!")
        print(f"   User ID: {existing_admin['_id']}")
        return
    
    # Hash the password
    hashed_password = pwd_context.hash(admin_data["password"])
    
    # Prepare user document
    admin_user = {
        "username": admin_data["username"],
        "email": admin_data["email"],
        "hashed_password": hashed_password,
        "role": admin_data["role"],
        "is_indigenous": admin_data["is_indigenous"],
        "onc_token": admin_data["onc_token"]
    }
    
    # Insert admin user
    try:
        result = await users_collection.insert_one(admin_user)
        print("✅ Admin account created successfully!")
        print(f"   Username: {admin_data['username']}")
        print(f"   Email: {admin_data['email']}")
        print(f"   Password: {admin_data['password']} (CHANGE THIS!)")
        print(f"   Role: {admin_data['role']}")
        print(f"   User ID: {result.inserted_id}")
        print()
        print("🔐 IMPORTANT: Change the admin password after first login!")
        
    except Exception as e:
        print(f"❌ Error creating admin account: {e}")

async def create_custom_admin():
    """Create a custom admin account with user input"""
    print("\n🔧 Create Custom Admin Account")
    print("-" * 40)
    
    username = input("Enter admin username (default: admin): ").strip() or "admin"
    email = input("Enter admin email (default: admin@oncai.com): ").strip() or "admin@oncai.com"
    password = input("Enter admin password (default: admin123): ").strip() or "admin123"
    
    # Check if user already exists
    existing_user = await users_collection.find_one({"username": username})
    if existing_user:
        print(f"❌ User '{username}' already exists!")
        return
    
    existing_email = await users_collection.find_one({"email": email})
    if existing_email:
        print(f"❌ Email '{email}' already exists!")
        return
    
    # Hash the password
    hashed_password = pwd_context.hash(password)
    
    # Prepare user document
    admin_user = {
        "username": username,
        "email": email,
        "hashed_password": hashed_password,
        "role": "admin",
        "is_indigenous": False,
        "onc_token": "admin-dev-token"
    }
    
    # Insert admin user
    try:
        result = await users_collection.insert_one(admin_user)
        print("\n✅ Custom admin account created successfully!")
        print(f"   Username: {username}")
        print(f"   Email: {email}")
        print(f"   Password: {password}")
        print(f"   Role: admin")
        print(f"   User ID: {result.inserted_id}")
        
    except Exception as e:
        print(f"❌ Error creating admin account: {e}")

async def main():
    print("🛠️  ONC AI Assistant - Admin Account Creator")
    print("=" * 50)
    
    print("\nChoose an option:")
    print("1. Create default admin account (username: admin, password: admin123)")
    print("2. Create custom admin account")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        await create_admin_account()
    elif choice == "2":
        await create_custom_admin()
    elif choice == "3":
        print("👋 Goodbye!")
        return
    else:
        print("❌ Invalid choice. Please try again.")
        await main()

if __name__ == "__main__":
    asyncio.run(main())
