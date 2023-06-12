import React, { useState } from "react";

const BASE_URL = "http://localhost:5000";

const App = () => {
  const [searchType, setSearchType] = useState("item_name");
  const [searchTerm, setSearchTerm] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSearchTypeChange = (event) => {
    setSearchType(event.target.value);
    setSearchResults([]);
  };

  const handleSearchTermChange = (event) => {
    setSearchTerm(event.target.value);
  };

  const handleSearch = () => {
    if (searchTerm === "") {
      setSearchResults([]);
      return;
    }

    setLoading(true);

    let apiUrl;
    if (searchType === "item_name") {
      apiUrl = `${BASE_URL}/item/search?query=${encodeURIComponent(
        searchTerm
      )}`;
    } else if (searchType === "category") {
      apiUrl = `${BASE_URL}/category/search?query=${encodeURIComponent(
        searchTerm
      )}`;
    } else if (searchType === "seller") {
      apiUrl = `${BASE_URL}/seller/search?query=${encodeURIComponent(
        searchTerm
      )}`;
    }

    fetch(apiUrl)
      .then((response) => response.json())
      .then((data) => {
        const sortedData = data.sort((a, b) => b.similarity - a.similarity);
        console.log(sortedData);
        setSearchResults(sortedData);
        setLoading(false);
      })
      .catch((error) => {
        console.error("Error:", error);
        setLoading(false);
      });
  };

  const renderItemsTable = () => {
    return (
      <table className="w-full bg-white shadow-lg">
        <thead>
          <tr className="bg-blue-200">
            <th className="py-4 px-6 text-left text-blue-800 font-semibold">
              Item Name
            </th>
            <th className="py-4 px-6 text-left text-blue-800 font-semibold">
              Category
            </th>
            <th className="py-4 px-6 text-left text-blue-800 font-semibold">
              Similarity
            </th>
          </tr>
        </thead>
        <tbody>
          {searchResults.map((item, index) => (
            <tr
              key={index}
              className={index % 2 === 0 ? "bg-blue-50" : "bg-white"}
            >
              <td className="py-4 px-6 text-blue-800">{item.item_name}</td>
              <td className="py-4 px-6 text-blue-800">{item.category}</td>
              <td className="py-4 px-6 text-blue-800">{item.similarity}</td>
            </tr>
          ))}
        </tbody>
      </table>
    );
  };

  const renderCategoriesTable = () => {
    return (
      <table className="w-full bg-white shadow-lg">
        <thead>
          <tr className="bg-blue-200">
            <th className="py-4 px-6 text-left text-blue-800 font-semibold">
              Category
            </th>
            <th className="py-4 px-6 text-left text-blue-800 font-semibold">
              Similarity
            </th>
          </tr>
        </thead>
        <tbody>
          {searchResults.map((category, index) => (
            <tr
              key={index}
              className={index % 2 === 0 ? "bg-blue-50" : "bg-white"}
            >
              <td className="py-4 px-6 text-blue-800">
                {category.category_name}
              </td>
              <td className="py-4 px-6 text-blue-800">{category.similarity}</td>
            </tr>
          ))}
        </tbody>
      </table>
    );
  };

  const renderSellersTable = () => {
    return (
      <table className="w-full bg-white shadow-lg">
        <thead>
          <tr className="bg-blue-200">
            <th className="py-4 px-6 text-left text-blue-800 font-semibold">
              Seller
            </th>
            <th className="py-4 px-6 text-left text-blue-800 font-semibold">
              Similarity
            </th>
          </tr>
        </thead>
        <tbody>
          {searchResults.map((seller, index) => (
            <tr
              key={index}
              className={index % 2 === 0 ? "bg-blue-50" : "bg-white"}
            >
              <td className="py-4 px-6 text-blue-800">{seller.seller_name}</td>
              <td className="py-4 px-6 text-blue-800">{seller.similarity}</td>
            </tr>
          ))}
        </tbody>
      </table>
    );
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-4xl font-bold text-blue-800 mb-8 text-center">
        Semantic Search
      </h1>

      <div className="flex mb-8">
        <select
          value={searchType}
          onChange={handleSearchTypeChange}
          className="w-48 rounded-l-lg py-3 px-4 bg-blue-100 text-blue-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="item_name">Item Name</option>
          <option value="category">Category</option>
          <option value="seller">Seller</option>
        </select>

        <input
          type="text"
          value={searchTerm}
          onChange={handleSearchTermChange}
          onKeyPress={(event) => {
            if (event.key === "Enter") {
              handleSearch();
            }
          }}
          className="flex-1 rounded-r-lg py-3 px-4 bg-blue-100 text-blue-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder="Search term..."
        />
      </div>

      {loading ? (
        <div className="flex justify-center items-center h-32">
          <div className="animate-spin rounded-full h-10 w-10 border-t-2 border-b-2 border-blue-800"></div>
        </div>
      ) : (
        <>
          {searchType === "item_name" && renderItemsTable()}
          {searchType === "category" && renderCategoriesTable()}
          {searchType === "seller" && renderSellersTable()}
        </>
      )}
    </div>
  );
};

export default App;
