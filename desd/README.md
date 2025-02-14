Web Application Development Plan
===============================

Tailwind build command: `docker-compose run tailwind npx tailwindcss -i ./myapp/static/src/input.css -o ./myapp/static/src/output.css --watch`

The web application is expected to meet the following requirements:

1. At least the following pages are expected to be included:
	* The main (home) page letting users select the type of user and action ahead;
	* Login/registration page for end-users is required,
	* A Dashboard page for end users (customers),
	* A Dashboard page for AI developers,
	* A Dashboard page for finance team members,
	* A Dashboard page for “admin” users that lets to process the operations as required.
2. A user should be able to navigate through the pages, smoothly, and especially be able to access to its own dashboard and the home page from any page.
3. A member should remain logged in until either the session is timeout or user is changed/logged out. With this respect, access to any different member’s dashboard must be secured.
4. The database is required to be separately deployed and connected via Docker containers. If a similar is work done for Advanced Databases module, that may be reused.
5. The ‘Machine Learning As A Service’ can be deployed on another separate container and be accessed via web/restful services.
6. The whole system should be using:
	* Python Django following MVC (MVT) patterns,
	* A Database system, (e.g., MySQL, PostgreSQL etc.,) should be separately deployed and accessed for handling data,
	* Deployment should be conducted via Docker containers,
	* Django Restful Framework (DRF) should be used together with Docker for creating, deploying, and accessing the services.

Members
-------

* Brody Wilson
* Hamid Rahman
* James Smith
* Kacper Drupisz
* Micheal Gamston

Development Plan
---------------

### Sprint 1 (Week 1-2)

| Task | Description | Assigned To |
| --- | --- | --- |
| Set up project structure and environment (Python Django, Docker) some SQL| Set up the project structure and environment using Python Django and Docker | Hamid |
| Develop the main page with user selection options  | Develop the main page with user selection options | Micheal |
| Develop the login/registration page for end-users | Develop the login/registration page for end-users | Brody |
| Implement user authentication and authorization | Implement user authentication and authorization | James |
| Users pages Admin, maintaining and user | Implement user authentication and authorization | Kacper |

### Sprint 2 (Week 3-4)

| Task | Description | Assigned To |
| --- | --- | --- |
| Develop the dashboard pages for end users, AI developers, finance team members, and admin users | Develop the dashboard pages for end users, AI developers, finance team members, and admin users |  |
| Implement navigation between pages and ensure that users can access their own dashboard and the home page from any page | Implement navigation between pages and ensure that users can access their own dashboard and the home page from any page |  |
| Ensure that users remain logged in until the session times out or the user is changed/logged out | Ensure that users remain logged in until the session times out or the user is changed/logged out |  |

### Sprint 3 (Week 5-6)

| Task | Description | Assigned To |
| --- | --- | --- |
| Deploy the database separately via Docker containers and connect it to the web application | Deploy the database separately via Docker containers and connect it to the web application |  |
| Develop the Machine Learning as a Service component and deploy it on another separate container | Develop the Machine Learning as a Service component and deploy it on another separate container |  |
| Integrate the Machine Learning as a Service component with the web application | Integrate the Machine Learning as a Service component with the web application |  |

### Sprint 4 (Week 7-8)

| Task | Description | Assigned To |
| --- | --- | --- |
| Conduct testing and debugging of the entire system | Conduct testing and debugging of the entire system |  |
| Ensure that the system meets all requirements and acceptance criteria defined in Sprint 1 | Ensure that the system meets all requirements and acceptance  |
