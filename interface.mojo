from chatty_cathy import chatty_cathyLanguageModel
from input_token_output import TokenIO
from collections import List
import json

struct chatty_cathyInterface:
    """
    User interface for chatty_cathy v4 model
    Supports: chat, generation, emotion queries, memory recall
    """
    var model: chatty_cathyLanguageModel
    var conversation_history: List[String]
    var is_loaded: Bool
    
    fn __init__(inout self, vocab_size: Int = 50000, embed_dim: Int = 256):
        self.model = chatty_cathyLanguageModel(vocab_size, embed_dim)
        self.conversation_history = List[String]()
        self.is_loaded = False
        print("chatty_cathy v4 initialized (untrained)")
    
    fn load_model(inout self, model_path: String, vocab_path: String):
        """Load trained model and tokenizer"""
        print("Loading chatty_cathy v4 from:", model_path)
        # TODO: Implement weight loading
        # self.model.model.embed.weight.load(model_path + "/embed.bin")
        # self.model.model.head.weight.load(model_path + "/head.bin")
        self.model.tokenizer.load_vocab(vocab_path)
        self.is_loaded = True
        print("✓ Model loaded successfully")
    
    fn save_model(self, model_path: String, vocab_path: String):
        """Save trained model and tokenizer"""
        print("Saving chatty_cathy v4 to:", model_path)
        # TODO: Implement weight saving
        # self.model.model.embed.weight.save(model_path + "/embed.bin")
        # self.model.model.head.weight.save(model_path + "/head.bin")
        self.model.tokenizer.save_vocab(vocab_path)
        print("✓ Model saved successfully")
    
    fn chat(inout self, user_input: String, max_length: Int = 100) -> String:
        """
        Interactive chat with chatty_cathy
        Maintains conversation history
        """
        if not self.is_loaded:
            print("⚠ Warning: Using untrained model")
        
        # Add to conversation history
        self.conversation_history.append("User: " + user_input)
        
        # Generate response with context
        var context = self._build_context()
        var response = self.model.generate(context + user_input, max_length, 0.85)
        
        self.conversation_history.append("chatty_cathy: " + response)
        
        return response
    
    fn generate(inout self, prompt: String, max_length: Int = 150, 
                temperature: Float32 = 0.9) -> String:
        """
        Generate text from prompt
        No conversation context
        """
        return self.model.generate(prompt, max_length, temperature)
    
    fn generate_fashion_scene(inout self, clothing_item: String) -> String:
        """Generate a fashion/beauty scene"""
        var prompt = f"She selected her {clothing_item} carefully. The fabric was"
        return self.model.generate(prompt, 200, 0.9)
    
    fn generate_romance_scene(inout self, emotion: String) -> String:
        """Generate romance scene with specific emotion"""
        # Set chatty_cathy's emotional state
        if emotion == "desire":
            self.model.cognition.emotional_state.subconscious_emotions["desire"] = 0.9
        elif emotion == "confidence":
            self.model.cognition.emotional_state.conscious_emotions["confidence"] = 0.9
        
        var prompt = "She felt a wave of " + emotion + " wash over her."
        return self.model.generate(prompt, 200, 0.85)
    
    fn query_emotions(self) -> Dict[String, Any]:
        """Get chatty_cathy's current emotional state"""
        return {
            "conscious": self.model.cognition.emotional_state.conscious_emotions,
            "subconscious": self.model.cognition.emotional_state.subconscious_emotions,
            "predicted_next": self.model.cognition.emotional_state.predict_next_emotion()
        }
    
    fn query_memories(self, topic: String) -> List[String]:
        """Recall memories about specific topic"""
        return self.model.cognition.memory.recall_relevant(topic)
    
    fn get_fashion_memories(self) -> List[String]:
        """Get all fashion/beauty memories"""
        return self.model.cognition.memory.fashion_memory
    
    fn set_personality(inout self, trait: String, value: Float32):
        """Adjust personality traits"""
        self.model.cognition.personality_traits[trait] = value
        print(f"Set {trait} to {value}")
    
    fn reset_conversation(inout self):
        """Clear conversation history"""
        self.conversation_history.clear()
        print("Conversation history cleared")
    
    fn _build_context(self) -> String:
        """Build context from recent conversation"""
        var context = ""
        var history_limit = 3  # Last 3 exchanges
        
        var start = max(0, len(self.conversation_history) - history_limit * 2)
        for i in range(start, len(self.conversation_history)):
            context += self.conversation_history[i] + "\n"
        
        return context


fn interactive_mode():
    """
    Interactive chat interface
    Run this for conversational mode
    """
    print("\n╔════════════════════════════════════════════════╗")
    print("║         chatty_cathy V4 - Interactive Mode           ║")
    print("╚════════════════════════════════════════════════╝\n")
    
    var chatty_cathy = chatty_cathyInterface(50000, 256)
    
    # Try to load trained model
    var model_exists = False  # Check if files exist
    if model_exists:
        chatty_cathy.load_model("./models/chatty_cathy_v4", "./models/vocab.json")
    else:
        print("⚠ No trained model found. Using untrained model.")
        print("  Train with: mojo training.mojo\n")
    
    print("Commands:")
    print("  /generate <prompt> - Generate text")
    print("  /fashion <item>    - Generate fashion scene")
    print("  /romance <emotion> - Generate romance scene")
    print("  /emotions          - Show emotional state")
    print("  /memories <topic>  - Recall memories")
    print("  /personality       - Show personality")
    print("  /reset             - Clear conversation")
    print("  /exit              - Quit\n")
    print("─" * 50 + "\n")
    
    while True:
        print("You: ", end="")
        var user_input = input()
        
        if user_input == "/exit":
            print("Goodbye! 💋")
            break
        
        elif user_input.startswith("/generate "):
            var prompt = user_input[10:]
            var output = chatty_cathy.generate(prompt, 150, 0.9)
            print("chatty_cathy:", output, "\n")
        
        elif user_input.startswith("/fashion "):
            var item = user_input[9:]
            var scene = chatty_cathy.generate_fashion_scene(item)
            print("chatty_cathy:", scene, "\n")
        
        elif user_input.startswith("/romance "):
            var emotion = user_input[9:]
            var scene = chatty_cathy.generate_romance_scene(emotion)
            print("chatty_cathy:", scene, "\n")
        
        elif user_input == "/emotions":
            var emotions = chatty_cathy.query_emotions()
            print("Emotional State:")
            print("  Conscious:", emotions["conscious"])
            print("  Subconscious:", emotions["subconscious"])
            print("  Predicted:", emotions["predicted_next"], "\n")
        
        elif user_input.startswith("/memories "):
            var topic = user_input[10:]
            var memories = chatty_cathy.query_memories(topic)
            print("Memories about", topic + ":")
            for memory in memories:
                print("  •", memory)
            print()
        
        elif user_input == "/personality":
            print("Personality Traits:")
            for trait, value in chatty_cathy.model.cognition.personality_traits.items():
                print(f"  {trait}: {value}")
            print()
        
        elif user_input == "/reset":
            chatty_cathy.reset_conversation()
            print()
        
        else:
            # Regular chat
            var response = chatty_cathy.chat(user_input, 100)
            print("chatty_cathy:", response, "\n")


fn api_mode():
    """
    API-style interface for programmatic use
    """
    print("\n╔════════════════════════════════════════════════╗")
    print("║           chatty_cathy V4 - API Mode                 ║")
    print("╚════════════════════════════════════════════════╝\n")
    
    var chatty_cathy = chatty_cathyInterface(50000, 256)
    chatty_cathy.load_model("./models/chatty_cathy_v4", "./models/vocab.json")
    
    # Example API calls
    print("=== API Examples ===\n")
    
    # 1. Generate fashion description
    print("1. Fashion Scene:")
    var fashion = chatty_cathy.generate_fashion_scene("red silk dress")
    print(fashion, "\n")
    
    # 2. Generate romance with emotion
    print("2. Romance Scene (desire):")
    var romance = chatty_cathy.generate_romance_scene("desire")
    print(romance, "\n")
    
    # 3. Check emotions
    print("3. Emotional State:")
    var emotions = chatty_cathy.query_emotions()
    print(emotions, "\n")
    
    # 4. Conversation
    print("4. Chat:")
    var chat1 = chatty_cathy.chat("Tell me about silk dresses")
    print("User: Tell me about silk dresses")
    print("chatty_cathy:", chat1, "\n")
    
    var chat2 = chatty_cathy.chat("What about makeup?")
    print("User: What about makeup?")
    print("chatty_cathy:", chat2, "\n")
    
    # 5. Adjust personality
    print("5. Personality Adjustment:")
    chatty_cathy.set_personality("confidence", 0.95)
    
    # 6. Fashion memories
    print("6. Fashion Memories:")
    var memories = chatty_cathy.get_fashion_memories()
    print("Stored", len(memories), "fashion memories\n")


fn batch_generation():
    """
    Batch generation mode for producing training data or content
    """
    print("\n╔════════════════════════════════════════════════╗")
    print("║         chatty_cathy V4 - Batch Generation           ║")
    print("╚════════════════════════════════════════════════╝\n")
    
    var chatty_cathy = chatty_cathyInterface(50000, 256)
    chatty_cathy.load_model("./models/chatty_cathy_v4", "./models/vocab.json")
    
    # Fashion items to generate scenes for
    var fashion_items = [
        "black evening gown",
        "red lipstick",
        "silk stockings",
        "pearl necklace",
        "lace lingerie",
        "designer heels",
        "perfume bottle",
        "diamond earrings"
    ]
    
    print("Generating fashion scenes...\n")
    
    for item in fashion_items:
        print("═" * 50)
        print("ITEM:", item)
        print("─" * 50)
        var scene = chatty_cathy.generate_fashion_scene(item)
        print(scene)
        print("\n")
    
    # Romance emotions
    var emotions = ["desire", "confidence", "lust", "joy"]
    
    print("\n" + "═" * 50)
    print("Generating romance scenes...\n")
    
    for emotion in emotions:
        print("═" * 50)
        print("EMOTION:", emotion)
        print("─" * 50)
        var scene = chatty_cathy.generate_romance_scene(emotion)
        print(scene)
        print("\n")


fn main():
    """
    Main entry point - choose mode
    """
    print("\n╔════════════════════════════════════════════════╗")
    print("║              chatty_cathy V4 INTERFACE               ║")
    print("║     Fashion • Beauty • Romance Generator      ║")
    print("╚════════════════════════════════════════════════╝\n")
    
    print("Select mode:")
    print("  1 - Interactive Chat")
    print("  2 - API Examples")
    print("  3 - Batch Generation")
    print("\nChoice (1-3): ", end="")
    
    var choice = input()
    
    if choice == "1":
        interactive_mode()
    elif choice == "2":
        api_mode()
    elif choice == "3":
        batch_generation()
    else:
        print("Invalid choice. Running interactive mode...")
        interactive_mode()


# Usage examples as standalone functions
fn example_simple_generation():
    """Simplest usage - just generate text"""
    var model = chatty_cathyInterface()
    var text = model.generate("She wore a beautiful dress", 100)
    print(text)


fn example_with_trained_model():
    """Load trained model and generate"""
    var model = chatty_cathyInterface()
    model.load_model("./models/chatty_cathy_v4", "./models/vocab.json")
    
    var text = model.generate("The perfume smelled of", 150, 0.85)
    print(text)


fn example_chat_session():
    """Multi-turn conversation"""
    var model = chatty_cathyInterface()
    model.load_model("./models/chatty_cathy_v4", "./models/vocab.json")
    
    var response1 = model.chat("What do you think about red dresses?")
    print("Response 1:", response1)
    
    var response2 = model.chat("How about makeup to match?")
    print("Response 2:", response2)
    
    var response3 = model.chat("Perfect! Tell me more.")
    print("Response 3:", response3)


fn example_emotional_generation():
    """Generate with specific emotional context"""
    var model = chatty_cathyInterface()
    model.load_model("./models/chatty_cathy_v4", "./models/vocab.json")
    
    # Check current emotional state
    var emotions = model.query_emotions()
    print("Current emotions:", emotions)
    
    # Generate romance scene
    var scene = model.generate_romance_scene("desire")
    print("Romance scene:", scene)
    
    # Check updated emotional state
    emotions = model.query_emotions()
    print("Updated emotions:", emotions)


fn example_memory_system():
    """Work with memory system"""
    var model = chatty_cathyInterface()
    
    # Generate some content to build memories
    model.generate_fashion_scene("silk dress")
    model.generate_fashion_scene("lipstick")
    
    # Recall fashion memories
    var fashion_memories = model.get_fashion_memories()
    print("Fashion memories:", len(fashion_memories))
    
    # Query specific topic
    var dress_memories = model.query_memories("dress")
    print("Dress-related memories:", dress_memories)


# Command-line interface
fn cli_main():
    """
    Command-line interface for quick operations
    Usage: mojo chatty_cathy_interface.mojo --mode chat
           mojo chatty_cathy_interface.mojo --generate "prompt here"
    """
    import sys
    
    var args = sys.argv()
    
    if len(args) < 2:
        print("Usage:")
        print("  mojo chatty_cathy_interface.mojo --chat")
        print("  mojo chatty_cathy_interface.mojo --generate 'prompt'")
        print("  mojo chatty_cathy_interface.mojo --batch")
        return
    
    var model = chatty_cathyInterface()
    
    # Try to load trained model
    try:
        model.load_model("./models/chatty_cathy_v4", "./models/vocab.json")
    except:
        print("⚠ No trained model found")
    
    if args[1] == "--chat":
        interactive_mode()
    elif args[1] == "--generate" and len(args) > 2:
        var prompt = args[2]
        var output = model.generate(prompt, 150)
        print(output)
    elif args[1] == "--batch":
        batch_generation()
    elif args[1] == "--api":
        api_mode()
    else:
        print("Unknown command:", args[1])