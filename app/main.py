from agent import run_agent

if __name__ == "__main__":
    user_id = input("Enter user id: ") #only for local testing 
    query = input("Enter your query: ")

    result = run_agent(query, user_id)

    print("\n📄 FINAL REPORT:\n")
    print(result)