import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

function SymptomsPage() {
  const navigate = useNavigate();
  const [symptoms, setSymptoms] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    
    // Simulate API call
    setTimeout(() => {
      console.log("Submitted symptoms:", symptoms);
      setIsSubmitting(false);
      navigate("/results"); // Navigate to results page after submission
    }, 1500);
  };

  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      {/* Navbar with back button */}
      <nav className="bg-orange-500 text-white px-4 py-4 shadow-md">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <button 
            onClick={() => navigate(-1)}
            className="text-white hover:text-gray-200 transition-colors"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
          </button>
          <h1 className="text-xl md:text-2xl font-bold">Describe Symptoms</h1>
          <div className="w-6"></div> {/* Spacer for balance */}
        </div>
      </nav>

      {/* Main content */}
      <main className="flex-grow flex flex-col items-center justify-center p-6">
        <div className="w-full max-w-md">
          <div className="bg-white p-6 rounded-xl shadow-lg">
            <h2 className="text-xl md:text-2xl font-semibold text-gray-800 mb-4 text-center">
              What symptoms are you experiencing?
            </h2>
            <p className="text-sm md:text-base text-gray-600 mb-6 text-center">
              Please describe your symptoms in detail to help with your assessment.
            </p>
            
            <form onSubmit={() => navigate("/results")}>
              <div className="mb-6">
                <textarea
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-orange-500 transition-colors"
                  rows="6"
                  placeholder="Example: I've had a persistent headache for 3 days, along with dizziness and nausea..."
                  value={symptoms}
                  onChange={(e) => setSymptoms(e.target.value)}
                  required
                />
              </div>
              
              <button
                type="submit"
                disabled={isSubmitting}
                className={`w-full bg-orange-500 text-white font-medium py-3 px-4 rounded-lg shadow transition-colors duration-300 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:ring-opacity-50 ${
                  isSubmitting ? 'opacity-70 cursor-not-allowed' : 'hover:bg-orange-600'
                }`}
              >
                {isSubmitting ? (
                  <span className="flex items-center justify-center">
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Processing...
                  </span>
                ) : (
                  "Submit Symptoms"
                )}
              </button>
            </form>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="p-4 text-center text-xs text-gray-500">
        <p>Be as detailed as possible for accurate assessment</p>
      </footer>
    </div>
  );
}

export default SymptomsPage;