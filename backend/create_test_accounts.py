"""
Quick Test Accounts Creator
Creates multiple test accounts for different roles quickly
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

# Test accounts to create
TEST_ACCOUNTS = [
    {
        "username": "generaluser",
        "email": "general@test.com",
        "password": "123456",
        "role": "general",
        "is_indigenous": False
    },
    {
        "username": "studentuser",
        "email": "student@test.com", 
        "password": "123456",
        "role": "student",
        "is_indigenous": False
    },
    {
        "username": "researcheruser",
        "email": "researcher@test.com",
        "password": "123456", 
        "role": "researcher",
        "is_indigenous": False
    },
    {
        "username": "educatoruser",
        "email": "educator@test.com",
        "password": "123456",
        "role": "educator", 
        "is_indigenous": False
    },
    {
        "username": "policyuser",
        "email": "policy@test.com",
        "password": "123456",
        "role": "policy-maker",
        "is_indigenous": False
    },
    {
        "username": "indigenoususer",
        "email": "indigenous@test.com",
        "password": "123456",
        "role": "general",
        "is_indigenous": True
    }
]

async def create_test_accounts():
    """Create all test accounts at once"""
    
    print("🚀 Creating Test Accounts")
    print("=" * 40)
    
    created_count = 0
    skipped_count = 0
    
    for account in TEST_ACCOUNTS:
        # Check if user already exists
        existing_user = await users_collection.find_one({"username": account["username"]})
        if existing_user:
            print(f"⏭️  Skipped: {account['username']} (already exists)")
            skipped_count += 1
            continue
            
        # Check if email already exists
        existing_email = await users_collection.find_one({"email": account["email"]})
        if existing_email:
            print(f"⏭️  Skipped: {account['username']} (email exists)")
            skipped_count += 1
            continue
        
        # Hash the password
        hashed_password = pwd_context.hash(account["password"])
        
        # Prepare user document
        user_doc = {
            "username": account["username"],
            "email": account["email"],
            "hashed_password": hashed_password,
            "role": account["role"],
            "is_indigenous": account["is_indigenous"],
            "onc_token": "test-dev-token"
        }
        
        try:
            result = await users_collection.insert_one(user_doc)
            indigenous_text = " (Indigenous)" if account["is_indigenous"] else ""
            print(f"✅ Created: {account['username']} - {account['role']}{indigenous_text}")
            created_count += 1
            
        except Exception as e:
            print(f"❌ Error creating {account['username']}: {e}")
    
    print("\n" + "=" * 40)
    print(f"📊 Summary: {created_count} created, {skipped_count} skipped")
    print("\n🔑 All passwords are: 123456")
    print("\n📝 Test Accounts Created:")
    print("-" * 25)
    for account in TEST_ACCOUNTS:
        indigenous_text = " (Indigenous)" if account["is_indigenous"] else ""
        print(f"   {account['username']} - {account['role']}{indigenous_text}")

async def show_all_users():
    """Display all users in the database"""
    print("\n👥 All Users in Database:")
    print("-" * 30)
    
    count = 0
    async for user in users_collection.find():
        indigenous_text = " (Indigenous)" if user.get("is_indigenous", False) else ""
        role = user.get("role", "unknown")
        username = user.get("username", "N/A")
        print(f"   {username} - {role}{indigenous_text}")
        count += 1
    
    print(f"\nTotal users: {count}")

async def main():
    print("⚡ Quick Test Account Creator")
    print("=" * 50)
    
    print("\nChoose an option:")
    print("1. Create all test accounts (generaluser, studentuser, etc.)")
    print("2. Show all existing users")
    print("3. Create accounts and show users")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        await create_test_accounts()
    elif choice == "2":
        await show_all_users()
    elif choice == "3":
        await create_test_accounts()
        await show_all_users()
    elif choice == "4":
        print("👋 Goodbye!")
        return
    else:
        print("❌ Invalid choice. Please try again.")
        await main()

if __name__ == "__main__":
    asyncio.run(main())
