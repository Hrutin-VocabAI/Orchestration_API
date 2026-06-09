import sys
import requests
import json

def run_tests():
    # Allow overriding server URL via CLI argument, default to localhost:5000
    server_base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"
    url_transcribe_endpoint = f"{server_base}/url_transcribe"

    test_links = [
        "https://libas-admin.c-zentrixcloud.com/playmediaAPI.php?agent_id=2023&session_id=1780913934.5361000000",
        "https://libas-admin.c-zentrixcloud.com/playmediaAPI.php?agent_id=2007&session_id=1780905450.3611000000",
        "https://libas-admin.c-zentrixcloud.com/playmediaAPI.php?agent_id=2021&session_id=1780912707.5072000000",
        "https://libas-admin.c-zentrixcloud.com/playmediaAPI.php?agent_id=2006&session_id=1780916287.6075000000",
        "https://libas-admin.c-zentrixcloud.com/playmediaAPI.php?agent_id=2006&session_id=1780917085.6332000000",
        "https://libas-admin.c-zentrixcloud.com/playmediaAPI.php?agent_id=2019&session_id=1780918981.6985000000"
    ]

    print(f"Target Server Endpoint: {url_transcribe_endpoint}")
    print("Sending 2 test requests to the endpoint...\n")

    # Send requests to the first 2 links as requested by the user
    for i, link in enumerate(test_links[:2]):
        print(f"--- TEST {i+1} ---")
        print(f"Input URL: {link}")
        
        payload = {
            "audio_url": link,
            "conversation_id": f"libas_test_conv_{i+1}"
        }
        
        try:
            r = requests.post(url_transcribe_endpoint, data=payload, timeout=120)
            print(f"Status Code: {r.status_code}")
            
            try:
                response_json = r.json()
                print("Response JSON (partial/keys):")
                # Print top-level keys first to verify the structure
                for key in sorted(response_json.keys()):
                    if key not in ["transcriptions", "agent_transcript", "customer_transcript", "alternatives", "utterances"]:
                        print(f"  {key}: {response_json[key]}")
                    else:
                        val = response_json[key]
                        summary_val = f"list with {len(val)} items" if isinstance(val, list) else f"str of length {len(val)}"
                        print(f"  {key}: <{summary_val}>")
                
                # Check for output fields
                if "URL" in response_json:
                    print(f"\nSUCCESS: Verified 'URL' is in response with value: {response_json['URL']}")
                else:
                    print("\nFAILURE: 'URL' field is missing from response")
            except Exception as parse_err:
                print(f"Failed to parse JSON response: {parse_err}")
                print(f"Raw Response: {r.text[:500]}...")
                
        except Exception as conn_err:
            print(f"Request failed: {conn_err}")
        print("\n" + "="*40 + "\n")

if __name__ == "__main__":
    run_tests()
