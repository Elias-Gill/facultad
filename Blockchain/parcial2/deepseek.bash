#!/bin/bash

# Configuración inicial
set -e
PROJECT_NAME="nft_deepseek"
FRONTEND_DIR="web_app"
CONTRACT_DIR="contracts"
BACKEND_DIR="backend"
ENV_FILE=".env"

# Crear estructura de directorios
mkdir -p $PROJECT_NAME
cd $PROJECT_NAME
mkdir -p $FRONTEND_DIR $CONTRACT_DIR $BACKEND_DIR

# Inicializar proyecto Hardhat
echo "Inicializando proyecto Hardhat..."
cd $CONTRACT_DIR
npm init -y > /dev/null
npm install --save-dev hardhat @nomicfoundation/hardhat-toolbox @openzeppelin/contracts @nomicfoundation/hardhat-verify dotenv > /dev/null
npx hardhat init > /dev/null <<< $'1\n'  # Seleccionar "Create a basic sample project"

# Configurar Hardhat
cat > hardhat.config.js <<EOF
require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config();

module.exports = {
  solidity: "0.8.20",
  networks: {
    ephemery: {
      url: "https://rpc.ephemery.dev",
      accounts: [process.env.PRIVATE_KEY || "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"]  // Clave privada de Hardhat (insegura, solo para testing)
  }
}
};
EOF

# Crear contrato Marketplace.sol
cat > contracts/Marketplace.sol <<EOF
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/utils/Counters.sol";

contract Marketplace is ERC721 {
    using Counters for Counters.Counter;
    Counters.Counter private _tokenIds;

    struct Listing {
        address owner;
        uint256 price;
        bool isSold;
    }

    mapping(uint256 => Listing) public listings;
    event ItemListed(uint256 indexed tokenId, address owner, uint256 price);
    event ItemSold(uint256 indexed tokenId, address buyer, uint256 price);

    constructor() ERC721("NFTMarketplace", "NFTM") {}

    function mintAndList(string memory _uri, uint256 _price) public {
        _tokenIds.increment();
        uint256 newTokenId = _tokenIds.current();
        _mint(msg.sender, newTokenId);
        listings[newTokenId] = Listing(msg.sender, _price, false);
        emit ItemListed(newTokenId, msg.sender, _price);
    }

    function buy(uint256 _tokenId) public payable {
        Listing storage listing = listings[_tokenId];
        require(!listing.isSold, "Item already sold");
        require(msg.value >= listing.price, "Insufficient funds");

        listing.isSold = true;
        _transfer(listing.owner, msg.sender, _tokenId);
        payable(listing.owner).transfer(msg.value);
        emit ItemSold(_tokenId, msg.sender, listing.price);
    }

    function getListing(uint256 _tokenId) public view returns (address, uint256, bool) {
        Listing memory listing = listings[_tokenId];
        return (listing.owner, listing.price, listing.isSold);
    }

    function withdraw() public {
        payable(msg.sender).transfer(address(this).balance);
    }
}
EOF

# Crear script de despliegue
cat > scripts/deploy.js <<EOF
const hre = require("hardhat");

async function main() {
    const Marketplace = await hre.ethers.getContractFactory("Marketplace");
    const marketplace = await Marketplace.deploy();
    await marketplace.waitForDeployment();
    console.log("Marketplace deployed to:", await marketplace.getAddress());
}

main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});
EOF

# Crear .env con valores por defecto (inseguros, solo para testing)
cd ..
cat > $ENV_FILE <<EOF
PRIVATE_KEY="0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
VITE_RPC_URL="http://localhost:8545"
VITE_CONTRACT_ADDRESS=""
EOF

# Configurar frontend con Vite + React
echo "Configurando frontend..."
cd $FRONTEND_DIR
npm create vite@latest . -- --template react > /dev/null
npm install ethers@5.7 react-router-dom @metamask/sdk > /dev/null

# Crear App.jsx básico
cat > src/App.jsx <<EOF
import { useState, useEffect } from 'react';
import { ethers } from 'ethers';
import './App.css';

function App() {
  const [account, setAccount] = useState('');
  const [nfts, setNfts] = useState([]);
  const [contract, setContract] = useState(null);

  useEffect(() => {
    init();
}, []);

  const init = async () => {
    if (window.ethereum) {
      try {
        const accounts = await window.ethereum.request({ method: 'eth_requestAccounts' });
        setAccount(accounts[0]);

        const provider = new ethers.BrowserProvider(window.ethereum);
        const signer = await provider.getSigner();
        const contractAddress = import.meta.env.VITE_CONTRACT_ADDRESS;
        const abi = [ /* ABI del contrato Marketplace */ ];
        const marketplace = new ethers.Contract(contractAddress, abi, signer);
        setContract(marketplace);

        loadMarketItems();
    } catch (error) {
        console.error(error);
    }
}
};

  const loadMarketItems = async () => {
    if (!contract) return;
    // Cargar 10 NFTs (simulados para demostración)
    const items = [];
    for (let i = 1; i <= 10; i++) {
      const [owner, price, isSold] = await contract.getListing(i);
      items.push({ id: i, owner, price: ethers.formatEther(price), isSold, uri: `https://example.com/nft/${i}.png` });
  }
    setNfts(items);
};

  const purchase = async (tokenId, price) => {
    if (!contract) return;
    try {
      const tx = await contract.buy(tokenId, { value: ethers.parseEther(price) });
      await tx.wait();
      loadMarketItems();
  } catch (error) {
      console.error(error);
  }
};

  return (
    <div className="App">
      <h1>NFT Marketplace</h1>
      <p>Connected account: {account}</p>
      <div className="nft-grid">
        {nfts.map(nft => (
          <div key={nft.id} className="nft-card">
            <img src={nft.uri} alt={`NFT ${nft.id}`} />
            <h3>NFT #{nft.id}</h3>
            <p>Price: {nft.price} ETH</p>
            <button onClick={() => purchase(nft.id, nft.price)} disabled={nft.isSold}>
              {nft.isSold ? 'Sold' : 'Buy'}
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
EOF

# Crear CSS básico
cat > src/App.css <<EOF
.App {
  text-align: center;
  padding: 20px;
}

.nft-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 20px;
  margin-top: 20px;
}

.nft-card {
  border: 1px solid #ddd;
  padding: 10px;
  border-radius: 8px;
}

.nft-card img {
  width: 100%;
  height: auto;
  border-radius: 4px;
}
EOF

# Volver al directorio raíz
cd ../..

# Crear Makefile
cat > Makefile <<EOF
.PHONY: all install deploy frontend test clean

all: install deploy frontend

install:
    cd contracts && npm install
    cd web_app && npm install

deploy:
    cd contracts && npx hardhat compile
    cd contracts && npx hardhat run scripts/deploy.js --network ephemery

frontend:
    cd web_app && npm run dev

test:
    cd contracts && npx hardhat test

clean:
    rm -rf node_modules
    rm -rf contracts/node_modules
    rm -rf web_app/node_modules
EOF

# Dar permisos de ejecución
chmod +x setup_project.sh

echo "¡Proyecto configurado con éxito!"
echo "Para comenzar:"
echo "1. Ejecuta 'make install' para instalar dependencias"
echo "2. Ejecuta 'make deploy' para desplegar el contrato"
echo "3. Actualiza VITE_CONTRACT_ADDRESS en .env con la dirección del contrato desplegado"
echo "4. Ejecuta 'make frontend' para iniciar el frontend"
