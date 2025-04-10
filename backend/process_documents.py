import os
import sys
import json
import argparse
import asyncio
from esg_document_processor import ESGDocumentProcessor

async def process_documents(files, output_dir, api_key=None, index=True, query=None):
    """
    Process multiple ESG documents and save results
    
    Args:
        files: List of file paths to process
        output_dir: Directory to save results
        api_key: OpenAI API key (optional)
        index: Whether to index documents in the vector database
        query: Optional query to focus the analysis
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize processor
    processor = ESGDocumentProcessor(openai_api_key="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")
    
    # Process each file
    results = []
    for file_path in files:
        print(f"Processing {file_path}...")
        result = await processor.process_document(file_path, index_document=index)
        results.append(result)
        
        # Save individual result
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        output_file = os.path.join(output_dir, f"{file_name}_analysis.json")
        processor.export_results(result, output_file)
        print(f"Results saved to {output_file}")
    
    # Save combined results
    combined_output = os.path.join(output_dir, "combined_analysis.json")
    processor.export_results(results, combined_output)
    print(f"Combined results saved to {combined_output}")
    
    # If query is provided, perform a search
    if query:
        print(f"\nPerforming search for query: '{query}'")
        query_result = await processor.query_documents(query)
        query_output = os.path.join(output_dir, "query_results.json")
        processor.export_results(query_result, query_output)
        print(f"Query results saved to {query_output}")
    
    return results

async def query_documents(query, output_file, api_key=None):
    """
    Query the vector database for relevant document chunks
    
    Args:
        query: Query text
        output_file: Path to output JSON file
        api_key: OpenAI API key (optional)
    """
    # Initialize processor
    processor = ESGDocumentProcessor(openai_api_key=api_key)
    
    # Perform query
    print(f"Querying documents with: '{query}'")
    result = await processor.query_documents(query)
    
    # Save results
    processor.export_results(result, output_file)
    print(f"Query results saved to {output_file}")
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Process ESG documents and generate analysis")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process documents")
    process_parser.add_argument("files", nargs="+", help="Paths to document files to process")
    process_parser.add_argument("--output", "-o", default="./results", help="Directory to save results")
    process_parser.add_argument("--api-key", help="OpenAI API key (defaults to OPENAI_API_KEY environment variable)")
    process_parser.add_argument("--no-index", action="store_true", help="Don't index documents in the vector database")
    process_parser.add_argument("--query", "-q", help="Optional query to focus the analysis")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query indexed documents")
    query_parser.add_argument("query", help="Query text")
    query_parser.add_argument("--output", "-o", default="./query_results.json", help="Path to output JSON file")
    query_parser.add_argument("--api-key", help="OpenAI API key (defaults to OPENAI_API_KEY environment variable)")
    
    args = parser.parse_args()
    
    if args.command == "process":
        # Validate files exist
        for file_path in args.files:
            if not os.path.exists(file_path):
                print(f"Error: File not found: {file_path}")
                sys.exit(1)
        
        # Process documents
        asyncio.run(process_documents(
            args.files, 
            args.output, 
            args.api_key, 
            not args.no_index,
            args.query
        ))
    
    elif args.command == "query":
        # Query documents
        asyncio.run(query_documents(
            args.query,
            args.output,
            args.api_key
        ))
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()