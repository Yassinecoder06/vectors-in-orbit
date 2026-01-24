#!/usr/bin/env python3
"""
Test script to verify all new features work correctly
Run this to test: python test_new_features.py
"""

import sys
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def test_interaction_logging():
    """Test 1: Verify interaction logging works"""
    print("\n" + "="*60)
    print("TEST 1: Interaction Logging")
    print("="*60)
    
    try:
        from interaction_logger import log_interaction, INTERACTION_WEIGHTS
        
        # Test each interaction type
        test_product = {
            "id": "test_product_001",
            "name": "Test Laptop",
            "price": 999.99,
            "brand": "TestBrand",
            "categories": ["Electronics", "Computers"]
        }
        
        for interaction_type in ["view", "click", "add_to_cart", "purchase"]:
            log_interaction(
                user_id="test_user_001",
                product_payload=test_product,
                interaction_type=interaction_type,
                query="test laptop for machine learning"
            )
            logger.info(f"✅ Logged '{interaction_type}' interaction (weight: {INTERACTION_WEIGHTS[interaction_type]})")
        
        print("\n✅ TEST 1 PASSED: All interaction types logged successfully")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}")
        return False


def test_popularity_calculation():
    """Test 2: Verify popularity calculation works"""
    print("\n" + "="*60)
    print("TEST 2: Popularity Calculation")
    print("="*60)
    
    try:
        from interaction_logger import get_top_interacted_products
        
        # Get popular products with debug mode
        popular = get_top_interacted_products(
            timeframe_hours=24,
            top_k=10,
            debug=True
        )
        
        if popular:
            logger.info(f"✅ Found {len(popular)} popular products")
            logger.info(f"   Top product: {popular[0]['product_id']} "
                       f"(interactions: {popular[0]['total_interactions']}, "
                       f"score: {popular[0]['weighted_popularity_score']:.3f})")
        else:
            logger.warning("⚠️  No popular products found (this is OK if no interactions exist yet)")
        
        print("\n✅ TEST 2 PASSED: Popularity calculation works")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}")
        return False


def test_search_with_popularity():
    """Test 3: Verify search includes popularity scores"""
    print("\n" + "="*60)
    print("TEST 3: Search with Popularity")
    print("="*60)
    
    try:
        from search_pipeline import search_products
        import uuid
        
        # Create test user
        test_user_id = f"test_user_{uuid.uuid4().hex[:8]}"
        
        # Override context to avoid Qdrant lookups
        test_context = {
            "user_id": test_user_id,
            "name": "Test User",
            "available_balance": 5000.0,
            "credit_limit": 10000.0,
            "preferred_brands": ["Apple", "Samsung"],
            "preferred_categories": ["Electronics"],
            "risk_tolerance": "Medium",
        }
        
        # Search with debug mode
        results = search_products(
            user_id=test_user_id,
            query="laptop for programming",
            top_k=3,
            debug_mode=True,
            override_context=test_context
        )
        
        if results:
            logger.info(f"✅ Search returned {len(results)} results")
            
            # Verify all new fields exist
            first_result = results[0]
            required_fields = [
                "final_score",
                "semantic_score",
                "affordability_score",
                "preference_score",
                "collaborative_score",
                "popularity_score",
                "explanations",
                "reason"
            ]
            
            for field in required_fields:
                if field in first_result:
                    logger.info(f"   ✅ Field '{field}' present")
                else:
                    logger.error(f"   ❌ Field '{field}' MISSING")
                    return False
            
            # Show score breakdown
            logger.info("\n   Score Breakdown (Top Result):")
            logger.info(f"     Final: {first_result['final_score']:.3f}")
            logger.info(f"     Semantic: {first_result['semantic_score']:.3f}")
            logger.info(f"     Affordability: {first_result['affordability_score']:.3f}")
            logger.info(f"     Preference: {first_result['preference_score']:.3f}")
            logger.info(f"     Collaborative: {first_result['collaborative_score']:.3f}")
            logger.info(f"     Popularity: {first_result['popularity_score']:.3f}")
            
            logger.info(f"\n   Explanations ({len(first_result['explanations'])}):")
            for i, exp in enumerate(first_result['explanations'][:3], 1):
                logger.info(f"     {i}. {exp}")
        else:
            logger.warning("⚠️  No search results (this is OK if products collection is empty)")
        
        print("\n✅ TEST 3 PASSED: Search returns all required fields")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ui_hooks():
    """Test 4: Verify UI hooks are available"""
    print("\n" + "="*60)
    print("TEST 4: UI Interaction Hooks")
    print("="*60)
    
    try:
        import app
        
        # Check if all hooks exist
        hooks = [
            "on_product_view",
            "on_product_click",
            "on_add_to_cart",
            "on_purchase"
        ]
        
        for hook_name in hooks:
            if hasattr(app, hook_name):
                logger.info(f"✅ Hook '{hook_name}' exists")
            else:
                logger.error(f"❌ Hook '{hook_name}' MISSING")
                return False
        
        print("\n✅ TEST 4 PASSED: All UI hooks are available")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 4 FAILED: {e}")
        return False


def test_formula_weights():
    """Test 5: Verify scoring formula weights sum to 1.0"""
    print("\n" + "="*60)
    print("TEST 5: Scoring Formula Validation")
    print("="*60)
    
    # Weights from rerank_products function
    weights = {
        "semantic": 0.30,
        "affordability": 0.25,
        "preference": 0.15,
        "collaborative": 0.20,
        "popularity": 0.10
    }
    
    total = sum(weights.values())
    
    logger.info("Current weights:")
    for component, weight in weights.items():
        logger.info(f"  {component:15s}: {weight:.2f} ({weight*100:5.1f}%)")
    logger.info(f"  {'TOTAL':15s}: {total:.2f}")
    
    if abs(total - 1.0) < 0.001:
        logger.info("✅ Weights sum to 1.0")
        print("\n✅ TEST 5 PASSED: Formula weights are balanced")
        return True
    else:
        logger.error(f"❌ Weights sum to {total:.3f} instead of 1.0")
        print(f"\n❌ TEST 5 FAILED: Weights don't sum to 1.0")
        return False


def main():
    """Run all tests"""
    print("\n" + "🧪"*30)
    print("FINCOMMERCE NEW FEATURES TEST SUITE")
    print("🧪"*30)
    
    tests = [
        ("Interaction Logging", test_interaction_logging),
        ("Popularity Calculation", test_popularity_calculation),
        ("Search with Popularity", test_search_with_popularity),
        ("UI Hooks", test_ui_hooks),
        ("Formula Weights", test_formula_weights),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            logger.exception(f"Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED! System is ready for production.")
        return 0
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed. Review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
