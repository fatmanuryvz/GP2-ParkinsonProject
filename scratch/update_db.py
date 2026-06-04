import os
from sqlalchemy import create_engine, Column, String, Boolean, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from werkzeug.security import generate_password_hash
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:1530@localhost:5432/neuroscan_db")
print(f"Connecting to database: {DATABASE_URL}")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(String(50), primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False)
    name = Column(String(100), nullable=False)
    is_approved = Column(Boolean, default=False)

def update_database():
    session = SessionLocal()
    try:
        # Check if users table exists
        with engine.connect() as conn:
            # Check table existence
            table_check = conn.execute(text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'users')"))
            exists = table_check.scalar()
            if not exists:
                print("Users table does not exist. Creating tables...")
                Base.metadata.create_all(bind=engine)
            
            # Check if is_approved column exists
            try:
                conn.execute(text("SELECT is_approved FROM users LIMIT 1"))
                print("is_approved column exists.")
            except Exception:
                print("is_approved column does not exist. Adding it...")
                conn.execute(text("ALTER TABLE users ADD COLUMN is_approved BOOLEAN DEFAULT TRUE"))
                print("is_approved column added successfully.")

        # Ensure admin user exists
        admin = session.query(User).filter(User.username == "admin").first()
        if admin:
            print(f"Admin user already exists (ID: {admin.id}, approved: {admin.is_approved}). Updating password...")
            admin.password_hash = generate_password_hash("admin123")
            admin.is_approved = True
            session.commit()
            print("Admin user updated successfully.")
        else:
            print("Admin user not found. Seeding admin user...")
            new_admin = User(
                id="A001",
                username="admin",
                password_hash=generate_password_hash("admin123"),
                role="admin",
                name="Sistem Yöneticisi",
                is_approved=True
            )
            session.add(new_admin)
            session.commit()
            print("Admin user created successfully.")

        # Print all users in database
        print("\nExisting Users in Database:")
        users = session.query(User).all()
        for u in users:
            print(f"- ID: {u.id}, Username: {u.username}, Role: {u.role}, Approved: {u.is_approved}")

    except Exception as e:
        session.rollback()
        print(f"Error during update: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    update_database()
