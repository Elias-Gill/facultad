#!/bin/bash

set -e

echo "Inicializando proyecto Hardhat + React..."

npm init -y > /dev/null
npm install --save-dev hardhat @nomicfoundation/hardhat-toolbox dotenv > /dev/null
npm install @openzeppelin/contracts > /dev/null

echo "Creando hardhat.config.js..."
cat > hardhat.config.js <<'EOF'
require("@nomicfoundation/hardhat-toolbox");

module.exports = {
  solidity: "0.8.20",
};
EOF

echo "Creando contrato Marketplace.sol..."
mkdir -p contracts
cat > contracts/Marketplace.sol <<'EOF'
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract Marketplace is ERC721URIStorage, Ownable {
    uint256 public nextTokenId;

    struct Listing {
        address owner;
        uint96 price;
        bool isSold;
    }

    mapping(uint256 => Listing) public listings;
    mapping(address => uint256) public balances;

    event ItemListed(uint256 tokenId, address seller, uint96 price);
    event ItemSold(uint256 tokenId, address buyer, uint96 price);

    constructor() ERC721("Parcial2NFT", "PNFT") {}

    function mintAndList(string memory _uri, uint96 _price) public {
        uint256 tokenId = nextTokenId++;
        _mint(msg.sender, tokenId);
        _setTokenURI(tokenId, _uri);
        listings[tokenId] = Listing(msg.sender, _price, false);
        emit ItemListed(tokenId, msg.sender, _price);
    }

    function buy(uint256 _tokenId) public payable {
        Listing storage item = listings[_tokenId];
        require(!item.isSold, "Item already sold");
        require(msg.value == item.price, "Incorrect ETH");

        item.isSold = true;
        balances[item.owner] += msg.value;
        _transfer(item.owner, msg.sender, _tokenId);

        emit ItemSold(_tokenId, msg.sender, item.price);
    }

    function getListing(uint256 _tokenId) public view returns (address, uint96, bool) {
        Listing memory item = listings[_tokenId];
        return (item.owner, item.price, item.isSold);
    }

    function withdraw() public {
        uint256 amount = balances[msg.sender];
        require(amount > 0, "No balance");
        balances[msg.sender] = 0;
        payable(msg.sender).transfer(amount);
    }
}
EOF

echo "Creando script deploy.js..."
mkdir -p scripts
cat > scripts/deploy.js <<'EOF'
async function main() {
  const [deployer] = await ethers.getSigners();
  const Marketplace = await ethers.getContractFactory("Marketplace");
  const contract = await Marketplace.deploy();
  await contract.deployed();
  console.log("Marketplace deployed to:", contract.address);
}
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
EOF

echo "Inicializando frontend con Vite + React..."
npm create vite@latest web_app -- --template react > /dev/null
cd web_app
npm install > /dev/null
npm install ethers@5.7 > /dev/null
cd ..

echo "Agregando frontend mínimo..."
cat > web_app/src/App.jsx <<'EOF'
import { useState } from 'react';
import { ethers } from 'ethers';

function App() {
  const [account, setAccount] = useState("");

  async function connectWallet() {
    if (window.ethereum) {
      const [addr] = await window.ethereum.request({ method: "eth_requestAccounts" });
      setAccount(addr);
    } else {
      alert("MetaMask no detectado");
    }
  }

  return (
    <div style={{ padding: 20 }}>
      <h1>Marketplace NFT - Parcial 2</h1>
      <button onClick={connectWallet}>Conectar MetaMask</button>
      {account && <p>Conectado: {account}</p>}
    </div>
  );
}

export default App;
EOF

echo "Agregando Makefile..."
cat > Makefile <<EOF
build:
\tnpx hardhat compile

deploy:
\tnpx hardhat run scripts/deploy.js

frontend:
\tcd web_app && npm run dev
EOF

echo "✅ Proyecto generado correctamente."
echo "Usá 'make build', 'make deploy', 'make frontend'"
