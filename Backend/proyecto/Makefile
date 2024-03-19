run:
	~/Library/wildfly-18/bin/standalone.sh

jboss-cli:
	~/Library/wildfly-18/bin/jboss-cli.sh

deploy:
	mvn clean install
	~/Library/wildfly-18/bin/standalone.sh

update:
	mvn clean install
	~/Library/wildfly-18/bin/jboss-cli.sh --connect --command="deploy ~/facultad/Backend/proyecto/target/prueba.war --force"

add-user:
	~/Library/wildfly-18/bin/add-user.sh

